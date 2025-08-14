from os import path
import pdb
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm

from .stimulus_analysis import StimulusAnalysis
from .proba_utils import get_chisq_response_proba

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class NaturalMovie(StimulusAnalysis):

    def __init__(self, session, plane, trace_type: str="events"):
        if trace_type not in ("dff", "events","cascade"):
            raise ValueError(f"{trace_type} must be either 'dff' or 'events'; given '{trace_type}'")
        
        stimulus = 'natural_movie'
        super().__init__(stimulus, stimulus, session, plane,trace_type)

        self.trace_type = trace_type
        self.frame_indices = np.array(sorted(self.stim_table["frame"].dropna().unique()), dtype=int) 
        self.stim_duration = 1/30   # duration of an individual stimulus (s)
        self.padding_duration = 0   # padding in between successive stimuli (s)
        self.n_repeats = 9          # number of movie repeats
        self.sig_p_thresh = 0.05
        self.frac_responsive_trials_thresh = 0.25 # Fraction of responses in the preferred direction that must be significant for the ROI to be flagged as responsive
        self.n_null_distribution_boot = 10000
        self.n_chisq_shuffles = 1000
        self.n_cv_iter = 20 # Number of cross-validation iterations
        
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
            self._trial_responses = xr.DataArray(
                data=np.nan,
                name="trial_responses",
                dims=("roi", "frame", "repeat"),
                coords=dict(
                    roi=self.session.get_rois(self.plane),
                    frame=self.frame_indices,
                    repeat=range(self.n_repeats)
                )
            )

            for imgID in self.frame_indices:
                stim_idx = self.get_stim_idx(imgID)
                self._trial_responses.loc[
                    dict(frame=imgID, repeat=range(len(stim_idx)))
                ] = self.sweep_responses[stim_idx, :].T # shape (n_rois, len(stim_idx))
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
            self._null_dist_multi_trial = self.get_spont_null_dist(self.baseline_time_window, self.response_time_window, n_boot=self.n_null_distribution_boot, n_means=self.n_repeats, trace_type=self.trace_type, cache=True)

        return self._null_dist_multi_trial

    def get_stim_idx(self, imgID):
        return self.stim_table.index[(self.stim_table["frame"] == imgID)]

    def _load_metrics(self):

        def ratio(p, q):
            return 0 if q == 0 else p/q

        all_metrics = []
        for roi in trange(self.n_rois):
            roi_trial_resp = self.trial_responses.sel(roi=roi)
            metrics = {}
            all_metrics.append(metrics)

            # Check if not valid
            if not self.is_roi_valid[roi]:
                continue

            # Check if bad ROI (i.e., it has all NaN responses)
            if np.all(np.isnan(roi_trial_resp)):
                print(f" (Skipping bad ROI {roi} with all NaN responses)", end="")
                continue

            # Compute preferred stimulus by taking argmax of mean responses
            roi_mean_trial_resp = np.nanmean(roi_trial_resp, axis=1)
            pref_img_idx = np.nanargmax(roi_mean_trial_resp)
            pref_response = roi_mean_trial_resp[pref_img_idx]
            pref_img = self.frame_indices[pref_img_idx]

            metrics['mean_responses'] = roi_mean_trial_resp
            metrics['pref_response'] = pref_response
            metrics['pref_img'] = pref_img
            metrics['pref_img_idx'] = pref_img_idx

            ## Z-score
            null_multi_mean = self.null_dist_multi_trial.mean(axis=1)
            null_multi_std = self.null_dist_multi_trial.std(axis=1)
            metrics["z_score"] = (pref_response - null_multi_mean[roi]) / null_multi_std[roi]

            metrics["z_score_responses"] = (roi_mean_trial_resp - null_multi_mean[roi]) / null_multi_std[roi]
            ## If mean response is above 95% null dist
            metrics["response_p"] = np.mean(pref_response < self._null_dist_multi_trial[roi])

            ## Determine if responsive
            pref_stim_trial_responses = self.trial_responses.sel(roi=roi, frame=pref_img_idx).values # ROI responses for preferred stimulus condition
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
            groups = []
            # groups.append(self._null_trial_responses[roi]) # Append baseline condition

            for iFrame in self.frame_indices:
                responses = self.trial_responses.sel(roi=roi, frame=iFrame) 
                responses = responses[~np.isnan(responses)] # remove nan entries
                groups.append(responses) # append stimulus condition responses

            _, p = st.f_oneway(*groups, axis=0)
            metrics["p_trial_responses"] = p
            metrics["sig_trial_responses"] = p < 0.05

            ## Normalized responses for each image
            norm_resp = []
            metrics["norm_responses"] = norm_resp
            mean_response = np.nanmean(self.trial_responses.sel(roi=roi))
            for iFrame in range(len(self.frame_indices)):
                resp = self.trial_responses.sel(roi=roi, frame=iFrame)
                mean_resp = np.nanmean(resp)
                norm_resp.append(mean_resp / mean_response)

        #Concatenate metrics aver cells
        metrics = pd.DataFrame(data=all_metrics)
        metrics["is_valid"] = self.is_roi_valid # ~metrics.isna().any(axis=1)
       
        # Is responsive using chi-squared test
        metrics["chisq_response_p"] = get_chisq_response_proba(self.stim_table, ["frame"], self._sweep_responses, n_shuffles=self.n_chisq_shuffles)

        # Null distribution
        metrics["null_dist_multi_mean"] = self._null_dist_multi_trial.mean(axis=1)
        metrics["null_dist_multi_std"] = self._null_dist_multi_trial.std(axis=1)
        metrics["null_dist_single_mean"] = self._null_dist_single_trial.mean(axis=1)
        metrics["null_dist_single_std"] = self._null_dist_single_trial.std(axis=1)

        metrics = metrics.convert_dtypes()
        self._metrics = metrics
