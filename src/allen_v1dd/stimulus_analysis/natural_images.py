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


class NaturalImages(StimulusAnalysis):

    def __init__(self, session, plane, ns_type: str="natural_images", trace_type: str="events", compute_chisq=True):
        if ns_type not in ("natural_images", "natural_images_12"):
            raise ValueError(f"ns_type must be either 'natural_images' or 'natural_images_12'; given '{ns_type}'")
        if trace_type not in ("dff", "events","cascade"):
            raise ValueError(f"{trace_type} must be either 'dff' or 'events'; given '{trace_type}'")
        
        super().__init__(ns_type, "ni12" if ns_type == "natural_images_12" else "ni", session, plane, trace_type)

        self.author = "David Wyrick, Chase King"
        self.ns_type = ns_type
        self.image_indices = np.array(sorted(self.stim_table["image_index"].dropna().unique()), dtype=int) 
        self.stim_duration = self.stim_meta["duration"] # duration of an individual stimulus (s)
        self.padding_duration = self.stim_meta["padding"] # padding in between successive stimuli (s)
        self.n_trials = int(np.mean(np.unique(self.stim_table["image_index"].dropna(),return_counts=True)[1]))  # number of trials that each stimulus is repeated
        self.sig_p_thresh = 0.05
        self.frac_responsive_trials_thresh = 0.25 # Fraction of responses in the preferred direction that must be significant for the ROI to be flagged as responsive
        self.n_null_distribution_boot = 10000
        self.n_chisq_shuffles = 1000 if compute_chisq else 0
        
        if trace_type == "dff":
            self.baseline_time_window = (-3, 0) # Time window used to offset/"demean" the event traces
            self.response_time_window = (0, self.stim_duration) # Time window used to compute stimulus response
        elif (trace_type == "events") or (trace_type == "cascade"):
            self.baseline_time_window = None # do not demean event traes
            self.response_time_window = (0, self.stim_duration) # Time window used to compute stimulus response
        
        self._metrics = None
        self._sweep_responses = None
        self._trial_responses = None
        self._null_trial_responses = None
        self._peak_response_metrics = {}
        self._stimulus_responses = {}
        self._null_dist_multi_trial = None
        self._null_dist_single_trial = None

    def save_to_h5(self, group):
        super().save_to_h5(group)

        group.attrs["ns_type"] = self.ns_type
        group.attrs["n_chisq_shuffles"] = self.n_chisq_shuffles

        # Trial responses
        ds = group.create_dataset("trial_responses", data=self.trial_responses)
        ds.attrs["dimensions"] = list(self.trial_responses.dims)

        for col in ("pref_response", "pref_img", "pref_img_idx", "z_score", "frac_responsive_trials", "lifetime_sparseness", "chisq_response_p"):
            if col in self.metrics.columns:
                group.create_dataset(col, data=self.metrics[col].values.astype(float))


    @property
    def sweep_responses(self):
        if self._sweep_responses is None:
            self._sweep_responses = np.zeros((len(self.stim_table), self.n_rois), dtype=float)
            for i in self.stim_table.index:
                start = self.stim_table.at[i, "start"]
                responses = self.get_responses(start, self.baseline_time_window, self.response_time_window)
                self._sweep_responses[i] = responses

        return self._sweep_responses
    
    @property
    def trial_responses(self):
        if self._trial_responses is None:
            self._trial_responses = xr.DataArray(
                data=np.nan,
                name="trial_responses",
                dims=("roi", "image", "trial"),
                coords=dict(
                    roi=self.session.get_rois(self.plane),
                    image=self.image_indices,
                    trial=range(self.n_trials)
                )
            )

            for image in self.image_indices:
                stim_idx = self.get_stim_idx(image)
                image_data = self.sweep_responses[stim_idx, :].T # shape (n_rois, n_frame_trials); note n_frame_trials not necessarily = n_trials
                self._trial_responses.loc[dict(image=image, trial=range(image_data.shape[1]))] = image_data
                # [:, image, :image_data.shape[1]] = image_data


        return self._trial_responses
    
    @property
    def metrics(self):
        if self._metrics is None:
            self._load_metrics()
        return self._metrics
    
    @property
    def null_dist_single_trial(self):
        if self._null_dist_single_trial is None:
            self._null_dist_single_trial = self.get_spont_null_dist(self.baseline_time_window, self.response_time_window, n_boot=self.n_null_distribution_boot, n_means=1, trace_type=self.trace_type, cache=True)

        return self._null_dist_single_trial
    
    @property
    def null_dist_multi_trial(self):
        if self._null_dist_multi_trial is None:
            self._null_dist_multi_trial = self.get_spont_null_dist(self.baseline_time_window, self.response_time_window, n_boot=self.n_null_distribution_boot, n_means=self.n_trials, trace_type=self.trace_type, cache=True)

        return self._null_dist_multi_trial

    def get_stim_idx(self, imgID):
        return self.stim_table.index[self.stim_table["image_index"] == imgID]

    def _load_metrics(self):
        all_metrics = []

        mean_trial_responses = self.trial_responses.mean(dim="trial", skipna=True)

        for roi in range(self.n_rois):
            
            metrics = {}
            all_metrics.append(metrics)

            # Check if not valid
            if not self.is_roi_valid[roi]:
                continue

            roi_trial_resp = self.trial_responses.sel(roi=roi)
            # Check if bad ROI (i.e., it has all NaN responses)
            if np.all(np.isnan(roi_trial_resp)):
                # print(f" (Skipping bad ROI {roi} with all NaN responses)", end="")
                continue

            # Compute preferred stimulus by taking argmax of mean responses
            roi_mean_trial_resp = mean_trial_responses.sel(roi=roi)
            argmax_dict = roi_mean_trial_resp.argmax(...)
            pref_img_idx = argmax_dict["image"].item()
            pref_img = self.image_indices[pref_img_idx]
            pref_response = roi_mean_trial_resp.isel(image=pref_img_idx).item()
            
            metrics['mean_responses'] = roi_mean_trial_resp.values
            metrics['pref_response'] = pref_response
            metrics['pref_img'] = pref_img
            metrics['pref_img_idx'] = pref_img_idx
            # import pdb; pdb.set_trace()
            ## Z-score
            null_mean = self.null_dist_multi_trial[roi].mean()
            null_std = np.std(self.null_dist_multi_trial[roi])
            metrics["z_score"] = (pref_response - null_mean) / null_std

            metrics["z_score_responses"] = (roi_mean_trial_resp.values - null_mean) / null_std
            ## If mean response is above 95% null dist
            metrics["response_p"] = np.mean(pref_response < self.null_dist_multi_trial[roi])

            ## Determine if responsive
            pref_stim_trial_responses = roi_trial_resp.sel(image=pref_img) # ROI responses for preferred stimulus condition
            pref_stim_trial_responses = pref_stim_trial_responses[~np.isnan(pref_stim_trial_responses)].values.reshape(-1, 1) # drop nans (no stimulus presentation) and make column vector
            p_values = np.mean(pref_stim_trial_responses < self.null_dist_single_trial[roi], axis=1) # p-values of responses when compared to null distribution
            # frac_sig = np.mean(np.logical_or(p_values < self.sig_p_thresh, p_values > (1-self.sig_p_thresh))) # fraction of p-values that are significant
            frac_sig = np.mean(p_values < self.sig_p_thresh) # fraction of p-values that are significant
            metrics["frac_responsive_trials"] = frac_sig
            # is_responsive = frac_sig >= self.frac_responsive_trials_thresh
            # metrics["is_responsive"] = is_responsive

            ## Compute lifetime sparseness
            # (See Olsen & Wilson 2008 for definition) (https://www.nature.com/articles/nature06864)
            responses = roi_trial_resp.values.reshape(-1)
            responses = responses[~np.isnan(responses)]
            lifetime_sparseness = (1 - (np.mean(responses)**2) / np.mean(np.square(responses))) / (1 - 1/len(responses))
            metrics["lifetime_sparseness"] = float(lifetime_sparseness)

            ## Compute p-values by comparing trial responses
            # VERY SLOW
            # groups = []
            # # groups.append(self._null_trial_responses[roi]) # Append baseline condition

            # for imgID in self.image_indices:
            #     responses = roi_trial_resp.sel(image=imgID)
            #     responses = responses[~np.isnan(responses)] # remove nan entries
            #     groups.append(responses) # append stimulus condition responses

            # _, p = st.f_oneway(*groups, axis=0)
            # metrics["p_trial_responses"] = p
            # metrics["sig_trial_responses"] = p < 0.05

            # ## Normalized responses for each image
            # norm_resp = []
            # metrics["norm_responses"] = norm_resp

            # mean_NS_response = roi_trial_resp.mean(skipna=True).item()
            # for imgID in self.image_indices:
            #     mean_response = self.trial_responses.sel(roi=roi,image=imgID).mean()
            #     norm_resp.append(mean_response / mean_NS_response)

        #Concatenate metrics aver cells
        metrics = pd.DataFrame(data=all_metrics)
        metrics["is_valid"] = self.is_roi_valid # ~metrics.isna().any(axis=1)
       
        # Is responsive using chi-squared test
        if self.n_chisq_shuffles > 0:
            metrics["chisq_response_p"] = get_chisq_response_proba(self.stim_table, ["image_index"], self._sweep_responses, n_shuffles=self.n_chisq_shuffles)

        # Null distribution
        metrics["null_dist_multi_mean"] = self.null_dist_multi_trial.mean(axis=1)
        metrics["null_dist_multi_std"] = np.std(self.null_dist_multi_trial, axis=1)

        metrics = metrics.convert_dtypes()
        self._metrics = metrics

