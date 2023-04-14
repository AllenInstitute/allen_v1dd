from os import path
import pdb
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt

from .stimulus_analysis import StimulusAnalysis
from .proba_utils import get_chisq_response_proba

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class NaturalMovie(StimulusAnalysis):
    def __init__(self, session, plane, trace_type: str="events", compute_chisq=True):
        if trace_type not in ("dff", "events","cascade"):
            raise ValueError(f"{trace_type} must be either 'dff' or 'events'; given '{trace_type}'")
        
        super().__init__("natural_movie", "nm", session, plane, trace_type)

        self.authors = "David Wyrick, Chase King"
        self.trace_type = trace_type
        self.frame_indices = np.array(sorted(self.stim_table["frame"].dropna().unique()), dtype=int) 
        self.stim_duration = 1/30   # duration of an individual stimulus (s)
        self.padding_duration = 0   # padding in between successive stimuli (s)
        self.n_repeats = self.stim_table.frame.value_counts().values[0] # Number of movie repeats
        self.sig_p_thresh = 0.05
        self.n_null_distribution_boot = 10000
        self.n_chisq_shuffles = 1000 if compute_chisq else 0
        
        if trace_type == "dff":
            self.baseline_time_window = (-1, 0) # Time window used to offset/"demean" the event traces
            self.response_time_window = (0, 4*self.time_per_frame) # Time window used to compute stimulus response
     
        elif (trace_type == "events") or (trace_type == "cascade"):
            self.baseline_time_window = None # do not demean event traes
            self.response_time_window = (0, 3*self.time_per_frame) # Time window used to compute stimulus response
        
        self._metrics = None
        self._sweep_responses = None
        self._trial_responses = None
        self._null_trial_responses = None
        self._peak_response_metrics = {}
        self._stimulus_responses = {}
        self._null_dist_multi_trial = None

    def save_to_h5(self, group):
        super().save_to_h5(group)

        group.attrs["n_frames"] = len(self.frame_indices)
        group.attrs["n_repeats"] = self.n_repeats
        group.attrs["frame_rate"] = 30
        group.attrs["n_chisq_shuffles"] = self.n_chisq_shuffles

        # Trial responses
        # DO NOT SAVE! THIS ARRAY IS TOO BIG!
        # ds = group.create_dataset("trial_responses", data=self.trial_responses)
        # ds.attrs["dimensions"] = list(self.trial_responses.dims)

        for col in ("pref_response", "pref_img", "pref_img_idx", "z_score", "frac_responsive_trials", "lifetime_sparseness", "chisq_response_p"):
            if col in self.metrics.columns:
                group.create_dataset(col, data=self.metrics[col].values.astype(float))


    @property
    def sweep_responses(self):
        if self._sweep_responses is None:
            ## Get responses to each stimulus
            self._sweep_responses = np.zeros((len(self.stim_table), self.n_rois), dtype=float)
            for i in self.stim_table.index:
                start = self.stim_table.at[i, "start"]
                self._sweep_responses[i] = self.get_responses(start, self.baseline_time_window, self.response_time_window)
        return self._sweep_responses
    
    @property
    def trial_responses(self):
        if self._trial_responses is None:
            data = np.full((self.n_rois, len(self.frame_indices), self.n_repeats), np.nan)

            for frame_i in self.frame_indices:
                stim_idx = self.get_stim_idx(frame_i)
                frame_data = self.sweep_responses[stim_idx, :].T # shape (n_rois, n_frame_repeats)
                data[:, frame_i, :frame_data.shape[1]] = frame_data

            self._trial_responses = xr.DataArray(
                data=data,
                name="trial_responses",
                dims=("roi", "frame", "repeat"),
                coords=dict(
                    roi=self.session.get_rois(self.plane),
                    frame=self.frame_indices,
                    repeat=range(self.n_repeats)
                )
            )

        return self._trial_responses

    @property
    def metrics(self):
        if self._metrics is None:
            self._load_metrics()
        return self._metrics
    
    @property
    def null_dist_multi_trial(self):
        if self._null_dist_multi_trial is None:
            self._null_dist_multi_trial = self.get_spont_null_dist(self.baseline_time_window, self.response_time_window, n_boot=self.n_null_distribution_boot, n_means=self.n_repeats, trace_type=self.trace_type, cache=True)

        return self._null_dist_multi_trial

    def get_stim_idx(self, imgID):
        return self.stim_table.index[self.stim_table["frame"] == imgID]

    def _load_metrics(self):
        all_metrics = []

        mean_trial_responses = self.trial_responses.mean(dim="repeat", skipna=True)

        for roi in range(self.n_rois):
            roi_trial_resp = self.trial_responses.sel(roi=roi)
            metrics = {}
            all_metrics.append(metrics)

            # Check if not valid
            if not self.is_roi_valid[roi]:
                continue

            # Check if bad ROI (i.e., it has all NaN responses)
            if np.all(np.isnan(roi_trial_resp.values)):
                print(f" (Skipping bad ROI {roi} with all NaN responses)", end="")
                continue

            # Compute preferred stimulus by taking argmax of mean responses
            roi_mean_trial_resp = mean_trial_responses.sel(roi=roi)
            pref_img_idx = int(np.nanargmax(roi_mean_trial_resp))
            pref_response = roi_mean_trial_resp.isel(frame=pref_img_idx).item()
            pref_img = self.frame_indices[pref_img_idx]

            metrics['mean_responses'] = roi_mean_trial_resp.values
            metrics['pref_response'] = pref_response
            metrics['pref_img'] = pref_img
            metrics['pref_img_idx'] = pref_img_idx

            ## Z-score
            null_mean = self.null_dist_multi_trial[roi].mean()
            null_std = np.std(self.null_dist_multi_trial[roi])
            metrics["z_score"] = (pref_response - null_mean) / null_std
            metrics["z_score_responses"] = (roi_mean_trial_resp.to_numpy() - null_mean) / null_std

            ## If mean response is above 95% null dist
            metrics["response_p"] = np.mean(pref_response < self.null_dist_multi_trial[roi])



            ## Determine if responsive
            pref_stim_trial_responses = roi_trial_resp.sel(frame=pref_img_idx).values # ROI responses for preferred stimulus condition
            pref_stim_trial_responses = pref_stim_trial_responses[~np.isnan(pref_stim_trial_responses)].reshape(-1, 1) # drop nans (no stimulus presentation) and make column vector
            
            # p_values = np.mean(pref_stim_trial_responses < self._null_dist_single_trial[roi], axis=1) # p-values of responses when compared to null distribution
            # # frac_sig = np.mean(np.logical_or(p_values < self.sig_p_thresh, p_values > (1-self.sig_p_thresh))) # fraction of p-values that are significant
            # frac_sig = np.mean(p_values < self.sig_p_thresh) # fraction of p-values that are significant
            # metrics["frac_responsive_trials"] = frac_sig
            
            #How many times is there a significant response during the preferred frame
            #any event is significant
            frac_responsive = np.sum(pref_stim_trial_responses > 0)/len(pref_stim_trial_responses)
            metrics["frac_responsive_trials"] = frac_responsive
            # is_responsive = frac_sig >= self.frac_responsive_trials_thresh
            # metrics["is_responsive"] = is_responsive

            ## Compute lifetime sparseness
            # (See Olsen & Wilson 2008 for definition) (https://www.nature.com/articles/nature06864)
            responses = roi_trial_resp.values.reshape(-1)
            responses = responses[~np.isnan(responses)]
            lifetime_sparseness = (1 - (np.mean(responses)**2) / np.mean(np.square(responses))) / (1 - 1/len(responses))
            metrics["lifetime_sparseness"] = float(lifetime_sparseness)

            ## Compute p-values by comparing trial responses
            # groups = []
            # # groups.append(self._null_trial_responses[roi]) # Append baseline condition

            # for iFrame in self.frame_indices:
            #     responses = roi_trial_resp.sel(frame=iFrame) 
            #     responses = responses[~np.isnan(responses)] # remove nan entries
            #     groups.append(responses) # append stimulus condition responses

            # _, p = st.f_oneway(*groups, axis=0)
            # metrics["p_trial_responses"] = p
            # metrics["sig_trial_responses"] = p < 0.05

            # Normalized responses for each image
            # norm_resp = []
            # metrics["norm_responses"] = norm_resp
            # mean_response = np.nanmean(self.trial_responses.sel(roi=roi))
            # for iFrame in range(len(self.frame_indices)):
            #     resp = self.trial_responses.sel(roi=roi, frame=iFrame)
            #     mean_resp = np.nanmean(resp)
            #     norm_resp.append(mean_resp / mean_response)

        #Concatenate metrics aver cells
        metrics = pd.DataFrame(data=all_metrics)
        metrics["is_valid"] = self.is_roi_valid # ~metrics.isna().any(axis=1)
       
        # Is responsive using chi-squared test
        if self.n_chisq_shuffles > 0:
            metrics["chisq_response_p"] = get_chisq_response_proba(self.stim_table, ["frame"], self.sweep_responses, n_shuffles=self.n_chisq_shuffles)

        # Null distribution
        metrics["null_dist_multi_mean"] = self.null_dist_multi_trial.mean(axis=1)
        metrics["null_dist_multi_std"] = np.std(self.null_dist_multi_trial, axis=1)

        metrics = metrics.convert_dtypes()
        self._metrics = metrics
