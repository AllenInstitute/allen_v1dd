import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt

from .stimulus_analysis import StimulusAnalysis
from .proba_utils import get_chisq_response_proba
from .fit_utils import vonmises_two_peak, vonmises_two_peak_fit, vonmises_two_peak_get_pref_dir_and_amplitude, r2_score, vonmises_two_peak_get_amplitude

def load_dg_xarray_from_h5(group, key):
    """Loads a drifting grating xarray from an h5 group.

    Args:
        group (hdf5 file group): Either a DGW or DGF group in the hdf5 file
        key (str): Key of array in group (e.g., "trial_responses")

    Returns:
        xarray.DataArray: xarray with labeled dimensions
    """
    dims = group[key].attrs["dimensions"]
    data = group[key][()]
    coords = {}

    for dim_name, dim_shape in zip(dims, data.shape):
        if dim_name == "roi":
            coords[dim_name] = range(dim_shape)
        elif dim_name == "direction":
            coords[dim_name] = group.attrs["directions"]
        elif dim_name == "spatial_frequency":
            coords[dim_name] = group.attrs["spatial_frequencies"]
        elif dim_name == "trial":
            # coords[dim_name] = range(group.attrs["n_trials"])
            coords[dim_name] = range(dim_shape)
    
    return xr.DataArray(data=data, dims=dims, coords=coords)

class DriftingGratings(StimulusAnalysis):
    """Used to analyze the drifting gratings stimulus.
    """

    def __init__(self, session, plane, trace_type: str="events", dg_type: str="full", quick_load=False, debug=True):
        if dg_type not in ("windowed", "full"):
            raise ValueError(f"dg_type must be either 'windowed' or 'full'; given '{dg_type}'")
        if trace_type not in ("dff", "events"):
            raise ValueError(f"{trace_type} must be either 'dff' or 'events'; given '{trace_type}'")

        super().__init__(f"drifting_gratings_{dg_type}", f"dg{dg_type[0]}", session, plane, trace_type)

        self.authors = "Chase King"
        self.dg_type = dg_type
        self.quick_load = quick_load
        self.debug = debug

        self.tf_list = np.array(sorted(self.stim_table["temporal_frequency"].dropna().unique()), dtype=int) # temporal frequency (Hz); in these data, only one TF is used (1 Hz)
        self.sf_list = np.array(sorted(self.stim_table["spatial_frequency"].dropna().unique()), dtype=float) # spatial frequency (cpd, cycles per degree)
        self.dir_list = np.array(sorted(self.stim_table["direction"].dropna().unique()), dtype=int) # direction (degrees; 0 = right and increases CCW) (?)
        self.contrast = self.stim_meta["contrast"] # in [0, 1] 
        self.stim_duration = self.stim_meta["duration"] # duration of an individual stimulus (s)
        self.padding_duration = self.stim_meta["padding"] # padding in between successive stimuli (s)
        self.center_position = self.stim_meta["center_position"] # (altitude, azimuth) in degrees of stimulus center
        self.radius = self.stim_meta["radius"] # radius (degrees) of stimulus
        self.n_trials = self.stim_meta["n_trials"] # number of trials that each stimulus is repeated
        # ^ note: not all stimuli are repeated this many times, so treat it as an upper bound
        
        self.sig_p_thresh = 0.05
        self.frac_responsive_trials_thresh = 0.5 # Fraction of responses in the preferred direction that must be significant for the ROI to be flagged as responsive
        self.n_null_distribution_boot = 10000
        self.n_chisq_shuffles = 100 if quick_load else 1000
        self.fit_tuning_curve = not quick_load # Whether to fit tuning curves to ROI responses
        
        if trace_type == "dff":
            self.baseline_time_window = (-3, 0) # Time window used to offset/"demean" the event traces
            self.response_time_window = (0, self.stim_duration) # Time window used to compute stimulus response
        elif trace_type == "events":
            self.baseline_time_window = None # do not demean event traes
            self.response_time_window = (0, self.stim_duration) # Time window used to compute stimulus response
        
        self._metrics = None
        self._sweep_responses = None
        self._trial_responses = None
        self._blank_trial_responses = None
        self._peak_response_metrics = {}
        self._stimulus_responses = {}
        self._null_dist_multi_trial = None
        self._null_dist_single_trial = None
        self._tuning_fit_params = None
        self._tuning_fit_metrics = None
        self._pref_cond_index = None

    def save_to_h5(self, group):
        super().save_to_h5(group)

        group.attrs["frac_sig_trials_thresh"] = self.frac_responsive_trials_thresh
        group.attrs["contrast"] = self.contrast
        group.attrs["directions"] = self.dir_list
        group.attrs["spatial_frequencies"] = self.sf_list
        group.attrs["temporal_frequencies"] = self.tf_list
        group.attrs["radius"] = self.radius
        group.attrs["n_trials"] = self.n_trials

        # Blank responses
        ds = group.create_dataset("blank_responses", data=self.blank_responses)
        ds.attrs["dimensions"] = ["roi", "blank_trial"]
        
        # Trial responses
        ds = group.create_dataset("trial_responses", data=self.trial_responses)
        ds.attrs["dimensions"] = list(self.trial_responses.dims)

        metrics = self.metrics

        # Used because sometimes metrics columns don't exist if there are no ROIs
        def get_met_col(col, default_val=0, dtype=float):
            if col in metrics.columns:
                return metrics[col].values.astype(dtype)
            else:
                return np.full(len(metrics), default_val, dtype=dtype)

        # Is responsive
        frac_resp = get_met_col("frac_responsive_trials")
        is_responsive = (metrics.is_valid & (frac_resp >= self.frac_responsive_trials_thresh)).values.astype(bool)
        ds = group.create_dataset("is_responsive", data=is_responsive)
        ds.attrs["inclusion_criteria"] = f"frac_responsive_trials >= {self.frac_responsive_trials_thresh}"
        group.create_dataset("frac_responsive_trials", data=frac_resp)

        # Preferred condition index
        ds = group.create_dataset("pref_cond_index", data=self.pref_cond_index)
        ds.attrs["dimensions"] = ["roi", "pref_cond_idx"]
        ds.attrs["notes"] = "Dimension 1 (pref_cond_idx) contains [pref_dir_idx, pref_sf_idx]"

        # Preferred condition
        pref_cond = np.full_like(self.pref_cond_index, np.nan, dtype=float)
        for roi in range(self.n_rois):
            pref_dir_idx, pref_sf_idx = self.pref_cond_index[roi]
            if pref_dir_idx >= 0 and pref_sf_idx >= 0:
                pref_cond[roi, 0] = self.dir_list[pref_dir_idx]
                pref_cond[roi, 1] = self.sf_list[pref_sf_idx]
        ds = group.create_dataset("pref_cond", data=pref_cond)
        ds.attrs["dimensions"] = ["roi", "pref_cond"]
        ds.attrs["notes"] = "Dimension 1 (pref_cond) contains [pref_dir, pref_sf]"

        # Trial running speeds
        trial_running_speeds = self.trial_running_speeds
        ds = group.create_dataset("trial_running_speeds", data=trial_running_speeds)
        ds.attrs["dimensions"] = list(trial_running_speeds.dims)

        # Various other metrics
        for col in ("dsi", "osi", "gosi", "pref_dir_mean", "lifetime_sparseness"):
            group.create_dataset(col, data=get_met_col(col, default_val=np.nan, dtype=float))

        # Tuning curves
        if self.fit_tuning_curve:
            ds = group.create_dataset("tuning_curve_params", data=self.tuning_fit_params)
            ds.attrs["dimensions"] = ["roi", "pref_sf_idx", "tuning_params"]
            ds.attrs["tuning_params"] = ["scale_1", "k_1", "x0", "scale_2", "k_2", "b"]

            for i, metric in enumerate(("pref_direction", "peak_amplitude", "r2")):
                ds = group.create_dataset(f"tuning_curve_{metric}", data=self.tuning_fit_metrics[:, :, i])
                ds.attrs["dimensions"] = ["roi", "pref_sf_idx", metric]

    @staticmethod
    def compute_ssi_from_h5(session, plane, plane_group, group_name="ssi"):
        def metric_index(a, b):
            if a + b == 0: return np.nan
            return (a - b) / (a + b)
        
        # If DG analyses are not in the file, then we can't compute anything
        if not ("drifting_gratings_windowed" in plane_group.keys() and "drifting_gratings_full" in plane_group.keys()):
            return

        dgw = plane_group["drifting_gratings_windowed"]
        dgf = plane_group["drifting_gratings_full"]
        dgw_trial_responses = dgw["trial_responses"][()]
        dgf_trial_responses = dgf["trial_responses"][()]
        n_rois = plane_group.attrs["n_rois"]
        valid_rois = np.where(plane_group["is_roi_valid"][()])[0]

        running_threshold = 1
        dgw_run = dgw["trial_running_speeds"][()]
        dgf_run = dgf["trial_running_speeds"][()]
        dgw_is_running = dgw_run > running_threshold
        dgw_is_stationary = dgw_run < running_threshold
        dgf_is_running = dgf_run > running_threshold
        dgf_is_stationary = dgf_run < running_threshold

        dgw_pref_cond_idxs = dgw["pref_cond_index"][()]

        metrics_descs = {
            "ssi": "Computed from pref DGW condition and DGF response at that condition",
            "ssi_avg": "Computed from average DGW and DGF responses (across direction and spatial freq.)",
            "ssi_avg_at_pref_sf": "Computed from average DGW and DGF responses (across direction at preferred DGW spatial freq.)",
            "ssi_running_avg_at_pref_sf": "Computed from average DGW and DGF responses while mouse is running (>1 cm/s) (across direction at preferred DGW spatial freq.)",
            "ssi_stationary_avg_at_pref_sf": "Computed from average DGW and DGF responses while mouse is stationary (<1 cm/s) (across direction at preferred DGW spatial freq.)",
            "ssi_running": "Same as SSI, but while mouse is running (>1 cm/s). Must have at least 3 running trials in DGW and DGF.",
            "ssi_stationary": "Same as SSI, but while mouse is stationary (<1 cm/s). Must have at least 3 stationary trials in DGW and DGF.",
            "ssi_tuning_fit": "Computed from tuning curve fits. Uses DGW pref sf and corresponding tuning fit pref. dir. Then uses DGF tuning fit at that pref sf and dir."
        }

        metrics = {
            metric: np.full(n_rois, np.nan)
            for metric in metrics_descs.keys()
        }

        # Wrapper method to ignore "RuntimeWarning: Mean of empty slice" warnings
        def nanmean(a):
            return np.nan if (len(a) == 0 or np.all(np.isnan(a))) else np.nanmean(a)

        for roi in valid_rois:
            pref_dir_idx, pref_sf_idx = dgw_pref_cond_idxs[roi]

            if pref_dir_idx == -1 or pref_sf_idx == -1:
                continue

            # Normal computation
            pref_dgw_resp = nanmean(dgw_trial_responses[roi, pref_dir_idx, pref_sf_idx])
            dgf_resp_pref_dgw = nanmean(dgf_trial_responses[roi, pref_dir_idx, pref_sf_idx])
            metrics["ssi"][roi] = metric_index(pref_dgw_resp, dgf_resp_pref_dgw)

            # Average across dir and SF
            mean_dgw_resp = nanmean(dgw_trial_responses[roi])
            mean_dgf_resp = nanmean(dgf_trial_responses[roi])
            metrics["ssi_avg"][roi] = metric_index(mean_dgw_resp, mean_dgf_resp)

            # Average across dir at DGW pref SF
            sf_mean_dgw_resp = nanmean(dgw_trial_responses[roi, :, pref_sf_idx])
            sf_mean_dgf_resp = nanmean(dgf_trial_responses[roi, :, pref_sf_idx])
            metrics["ssi_avg_at_pref_sf"][roi] = metric_index(sf_mean_dgw_resp, sf_mean_dgf_resp)

            # Running avg at pref TF
            sf_mean_dgw_resp_running = nanmean(np.where(dgw_is_running, dgw_trial_responses, np.nan)[roi, :, pref_sf_idx]) # dgw.trial_responses.where(dgw_is_running).sel(roi=roi, spatial_frequency=dgw_pref_sf).mean().item()
            sf_mean_dgf_resp_running = nanmean(np.where(dgf_is_running, dgf_trial_responses, np.nan)[roi, :, pref_sf_idx]) # dgf.trial_responses.where(dgf_is_running).sel(roi=roi, spatial_frequency=dgw_pref_sf).mean().item()
            metrics["ssi_running_avg_at_pref_sf"][roi] = metric_index(sf_mean_dgw_resp_running, sf_mean_dgf_resp_running)

            sf_mean_dgw_resp_stationary = nanmean(np.where(dgw_is_stationary, dgw_trial_responses, np.nan)[roi, :, pref_sf_idx]) # dgw.trial_responses.where(dgw_is_stationary).sel(roi=roi, spatial_frequency=dgw_pref_sf).mean().item()
            sf_mean_dgf_resp_stationary = nanmean(np.where(dgf_is_stationary, dgf_trial_responses, np.nan)[roi, :, pref_sf_idx]) # dgf.trial_responses.where(dgf_is_stationary).sel(roi=roi, spatial_frequency=dgw_pref_sf).mean().item()
            metrics["ssi_stationary_avg_at_pref_sf"][roi] = metric_index(sf_mean_dgw_resp_stationary, sf_mean_dgf_resp_stationary)

            ws = np.where(dgw_is_stationary, dgw_trial_responses, np.nan)[roi, pref_dir_idx, pref_sf_idx]
            fs = np.where(dgf_is_stationary, dgf_trial_responses, np.nan)[roi, pref_dir_idx, pref_sf_idx]
            ws = ws[~np.isnan(ws)]
            fs = fs[~np.isnan(fs)]
            n_trials_required = 3
            if len(ws) >= n_trials_required and len(fs) >= n_trials_required:
                w = ws.mean()
                f = fs.mean()
                metrics["ssi_stationary"][roi] = metric_index(w, f)
            
            wr = np.where(dgw_is_running, dgw_trial_responses, np.nan)[roi, pref_dir_idx, pref_sf_idx]
            fr = np.where(dgf_is_running, dgf_trial_responses, np.nan)[roi, pref_dir_idx, pref_sf_idx]
            wr = wr[~np.isnan(wr)]
            fr = fr[~np.isnan(fr)]
            if len(wr) >= n_trials_required and len(fr) >= n_trials_required:
                w = wr.mean()
                f = fr.mean()
                metrics["ssi_running"][roi] = metric_index(w, f)

            # SSI from tuning curve fits
            k = "tuning_curve_params"
            if k in dgw.keys() and k in dgf.keys():
                dgw_params = dgw[k][roi, pref_sf_idx]
                dgf_params = dgf[k][roi, pref_sf_idx]

                if not np.any(np.isnan(dgw_params) | np.isnan(dgf_params)):
                    tuning_pref_dir, _ = vonmises_two_peak_get_pref_dir_and_amplitude(dgw_params)
                    w = vonmises_two_peak(tuning_pref_dir, *dgw_params)
                    f = vonmises_two_peak(tuning_pref_dir, *dgf_params)
                    metrics["ssi_tuning_fit"] = metric_index(w, f)
        
        # Save to an h5 group
        ssi_group = plane_group.create_group(group_name)
        
        for metric, values in metrics.items():
            ds = ssi_group.create_dataset(metric, data=values)
            ds.attrs["desc"] = metrics_descs[metric]

    @property
    def metrics(self):
        if self._metrics is None:
            self._load_responses()
        return self._metrics
    
    @property
    def trial_responses(self):
        if self._trial_responses is None:
            self._trial_responses = xr.DataArray(
                data=np.nan,
                name="trial_responses",
                dims=("roi", "direction", "spatial_frequency", "trial"), # Could include temporal frequency, but there is only 1 value (1)
                coords=dict(
                    roi=self.session.get_rois(self.plane),
                    direction=self.dir_list,
                    spatial_frequency=self.sf_list,
                    trial=range(self.n_trials)
                )
            )

            for dir in self.dir_list:
                for sf in self.sf_list:
                    stim_idx = self.get_stim_idx(dir, sf)
                    self._trial_responses.loc[
                        dict(direction=dir, spatial_frequency=sf, trial=range(len(stim_idx)))
                    ] = self.sweep_responses[stim_idx, :].T # shape (n_rois, len(stim_idx))

        return self._trial_responses
    
    @property
    def blank_responses(self):
        if self._blank_trial_responses is None:
            null_stim_idx = self.stim_table.index[self.stim_table.isna().any(axis=1)]
            self._blank_trial_responses = self.sweep_responses[null_stim_idx, :].T # shape (n_rois, len(null_stim_idx))

        return self._blank_trial_responses
    
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
    def trial_running_speeds(self):
        running_padding = 0.1

        running_speed = self.session.get_running_speed()
        trial_running_speeds = xr.DataArray(
            data=np.nan,
            name="trial_running_speeds",
            dims=("direction", "spatial_frequency", "trial"), # Could include temporal frequency, but there is only 1 value (1)
            coords=dict(
                direction=self.dir_list,
                spatial_frequency=self.sf_list,
                trial=range(self.n_trials)
            )
        )

        for dir in self.dir_list:
            for sf in self.sf_list:
                stim_idx = self.get_stim_idx(dir, sf)
                for trial, stim_i in enumerate(stim_idx):
                    start = self.stim_table.at[stim_i, "start"] - running_padding
                    end = self.stim_table.at[stim_i, "end"] + running_padding
                    mean_run = running_speed.sel(time=slice(start, end)).mean().item() # cm/s; mean running speed during stimulus sweep
                    trial_running_speeds.loc[dict(direction=dir, spatial_frequency=sf, trial=trial)] = mean_run

        return trial_running_speeds

    @property
    def tuning_fit_params(self):
        if self.fit_tuning_curve and self._tuning_fit_params is None:
            self._load_responses()
        return self._tuning_fit_params
    
    @property
    def tuning_fit_metrics(self):
        if self.fit_tuning_curve and self._tuning_fit_metrics is None:
            self._load_responses()
        return self._tuning_fit_metrics

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

    @property
    def pref_cond_index(self):
        if self._pref_cond_index is None:
            dg_mean_trial_resp = self.trial_responses.mean(dim="trial", skipna=True) # Trial-mean responses for each ROI
            argmax_dims = ("direction", "spatial_frequency")
            argmax_dict = dg_mean_trial_resp.fillna(-1).argmax(dim=argmax_dims) # { dim: array of argmax for each ROI }
            self._pref_cond_index = np.full((self.n_rois, 2), -1, dtype=int)
            
            for roi in range(self.n_rois):
                if self.is_roi_valid[roi]:
                    self._pref_cond_index[roi] = [argmax_dict[d][roi] for d in argmax_dims]
        
        return self._pref_cond_index


    def get_stim_idx(self, dir, sf):
        if dir is None and sf is None:
            return self.stim_table.index[self.stim_table.isna().any(axis=1)]

        dir_mask = self.stim_table["direction"] == dir
        sf_mask = self.stim_table["spatial_frequency"] == sf

        if dir is None:
            mask = sf_mask
        elif sf is None:
            mask = dir_mask
        else:
            mask = dir_mask & sf_mask

        return self.stim_table.index[mask]

    def _compute_osi(self, dir_tuning):
        # osi = 0 if pref_response == orth_response else (pref_response - orth_response) / (pref_response + orth_response)
        # Reference: https://www.frontiersin.org/articles/10.3389/fncir.2014.00092/full
        # dir_tuning = np.where(dir_tuning > 0, dir_tuning, 0) # TODO: Should tuning be rectified?
        L_norm = dir_tuning.sum().item() # Normalization factor
        L_ori = 0
        # (Not computing circular variance direction selectivity because it conflates orientation and direction selectivity)
        for dir_deg, resp in zip(self.dir_list, dir_tuning.values):
            # TODO: necessary to consider nans?
            dir_rad = np.deg2rad(dir_deg)
            L_ori += resp * np.exp(1j * 2 * dir_rad)
        if L_norm != 0: L_ori /= L_norm
        osi = np.abs(L_ori)
        return osi

    def _load_responses(self):
        # Load responses for each trial
        if self.debug: print(f"Loading DG-{self.dg_type} response metrics for session {self.session.get_session_id()}, plane {self.plane}...")

        null_multi_mean = self.null_dist_multi_trial.mean(axis=1)
        null_multi_std = self.null_dist_multi_trial.std(axis=1)
        
        # Load metrics
        # print("Computing quantitative ROI metrics...", end="")
        # metrics = pd.DataFrame(columns=["is_responsive", "pref_dir", "pref_sf", "pref_dir_idx", "pref_sf_idx", "osi", "dsi", "cir_var", "osi_cir_var"])
        all_metrics = []

        def ratio(p, q):
            return 0 if q == 0 else p/q
        
        self._tuning_fit_params = np.full((self.n_rois, len(self.sf_list), 6), np.nan) # n_params = 6 (scale_1, k_1, x0, scale_2, k_2, b)
        self._tuning_fit_metrics = np.full((self.n_rois, len(self.sf_list), 3), np.nan) # n_metrics = 3 (pref_direction, peak_amplitude, r2)

        # for roi in tqdm(range(self.n_rois), desc="Loading ROI responses"):
        for roi in range(self.n_rois):
            roi_trial_resp = self.trial_responses.sel(roi=roi)
            metrics = {}
            all_metrics.append(metrics)

            # Check if not valid
            if not self.is_roi_valid[roi]:
                continue

            # Check if bad ROI (i.e., it has all NaN responses)
            if np.all(np.isnan(roi_trial_resp)):
                if self.debug: print(f" (Skipping bad ROI {roi} with all NaN responses)", end="")
                continue

            # Compute preferred stimulus by taking argmax of mean responses
            roi_mean_trial_resp = roi_trial_resp.mean("trial", skipna=True)
            argmax_dict = roi_mean_trial_resp.argmax(...)
            pref_dir_idx = argmax_dict["direction"].item()
            pref_sf_idx = argmax_dict["spatial_frequency"].item()
            pref_response = roi_mean_trial_resp.isel(direction=pref_dir_idx, spatial_frequency=pref_sf_idx).item()
            pref_dir = self.dir_list[pref_dir_idx]
            pref_sf = self.sf_list[pref_sf_idx]
            metrics["pref_response"] = pref_response
            metrics["pref_dir"] = pref_dir
            metrics["pref_sf"] = pref_sf
            metrics["pref_dir_idx"] = pref_dir_idx
            metrics["pref_sf_idx"] = pref_sf_idx
            
            # Compute mean preferred direction
            # complex_vector = np.mean([np.exp(np.deg2rad(self.dir_list[i]) * 1j) for i in pref_dir_idx_cv])
            # metrics["pref_dir_mean"] = np.mod(np.rad2deg(np.angle(complex_vector)), 360)
            # metrics["pref_sf_mean"] = np.mean([self.sf_list[i] for i in pref_sf_idx_cv])
            vec = 0
            for dir_idx, dir_degrees in enumerate(self.dir_list):
                theta = np.deg2rad(dir_degrees)
                # compute sum of all responses with dir and sf
                r = roi_mean_trial_resp.isel(direction=dir_idx, spatial_frequency=pref_sf_idx).sum(skipna=True).item()
                vec += r * np.exp(theta * 1j)
            metrics["pref_dir_mean"] = np.angle(vec, deg=True) % 360
            # metrics["osi"] = osi_cv.mean()
            # metrics["dsi"] = dsi_cv.mean()

            # Z-score
            metrics["z_score"] = (pref_response - null_multi_mean[roi]) / null_multi_std[roi]

            # If mean response is above 95% null dist
            metrics["response_p"] = np.mean(pref_response < self.null_dist_multi_trial[roi])

            # Compute OSI and DSI
            null_idx = DriftingGratings.get_null_dir_index(pref_dir_idx)
            orth_1_idx, orth_2_idx = DriftingGratings.get_orth_dir_indices(pref_dir_idx)
            null_response = roi_mean_trial_resp.isel(direction=null_idx, spatial_frequency=pref_sf_idx).item()
            orth_response = (roi_mean_trial_resp.isel(direction=orth_1_idx, spatial_frequency=pref_sf_idx).item()
                + roi_mean_trial_resp.isel(direction=orth_2_idx, spatial_frequency=pref_sf_idx).item()) / 2
            metrics["osi"] = ratio(pref_response - orth_response, pref_response + orth_response)
            metrics["gosi"] = self._compute_osi(roi_mean_trial_resp.isel(spatial_frequency=pref_sf_idx))
            metrics["dsi"] = ratio(pref_response - null_response, pref_response + null_response)
            # metrics["cir_var"] = 1 - osi_cir_var # 1 <=> no preference; 0 <=> perfect preference
            
            # Determine if responsive
            pref_stim_trial_responses = roi_trial_resp.isel(direction=pref_dir_idx, spatial_frequency=pref_sf_idx) # ROI responses for preferred stimulus condition
            pref_stim_trial_responses = pref_stim_trial_responses.dropna("trial").values.reshape(-1, 1) # drop nans (no stimulus presentation) and make column vector
            p_values = np.mean(pref_stim_trial_responses < self.null_dist_single_trial[roi], axis=1) # p-values of responses when compared to null distribution
            # frac_sig = np.mean(np.logical_or(p_values < self.sig_p_thresh, p_values > (1-self.sig_p_thresh))) # fraction of p-values that are significant
            frac_sig = np.mean(p_values < self.sig_p_thresh) # fraction of p-values that are significant
            metrics["frac_responsive_trials"] = frac_sig
            # is_responsive = frac_sig >= self.frac_responsive_trials_thresh
            # metrics["is_responsive"] = is_responsive

            # Compute lifetime sparseness
            # (See Olsen & Wilson 2008 for definition) (https://www.nature.com/articles/nature06864)
            responses = roi_trial_resp.values.reshape(-1)
            responses = responses[~np.isnan(responses)]
            # n = len(responses)
            # lifetime_sparseness = (1 - (np.square(np.sum(responses / n)) / (np.sum(np.square(responses) / n)))) / (1 - 1/n)
            lifetime_sparseness = (1 - (np.mean(responses)**2) / np.mean(np.square(responses))) / (1 - 1/len(responses))
            metrics["lifetime_sparseness"] = float(lifetime_sparseness)

            # Compute p-values by comparing trial responses
            groups = []
            groups.append(self.blank_responses[roi]) # Append baseline condition

            # Append stimulus condition responses
            for dir in self.trial_responses.direction:
                for sf in self.trial_responses.spatial_frequency:
                    groups.append(self.trial_responses.sel(roi=roi, direction=dir, spatial_frequency=sf).dropna("trial"))

            _, p = st.f_oneway(*groups, axis=0)
            metrics["p_trial_responses"] = p
            metrics["sig_trial_responses"] = p < 0.05

            # Fit tuning curve
            if self.fit_tuning_curve:
                for sf_idx in range(len(self.sf_list)):
                    x = self.dir_list
                    y = roi_mean_trial_resp.isel(spatial_frequency=sf_idx)
                    vonmises_params = vonmises_two_peak_fit(x, y) # scale_1, k_1, x0, scale_2, k_2, b
                    
                    if vonmises_params is not None:
                        tuning_dir, tuning_amp = vonmises_two_peak_get_pref_dir_and_amplitude(vonmises_params)
                        y_pred = vonmises_two_peak(x, *vonmises_params)
                        r2 = r2_score(y, y_pred)

                        self._tuning_fit_params[roi, sf_idx, :] = vonmises_params
                        self._tuning_fit_metrics[roi, sf_idx, :] = [tuning_dir, tuning_amp, r2]

                        # Fit was successful
                        # for param_i, param_name in enumerate(("s1", "k1", "x0", "s2", "k2", "b")):
                            # metrics[f"vonmises_param_{param_name}"] = vonmises_params[param_i]
                        
                        # Actually might be easier to interact with param_0, param_1, etc...
                        for param_i in range(len(vonmises_params)):
                            metrics[f"vonmises_{sf_idx}_param_{param_i}"] = vonmises_params[param_i]

                        metrics[f"vonmises_{sf_idx}_pref_dir"] = tuning_dir
                        metrics[f"vonmises_{sf_idx}_peak_amp"] = tuning_amp
                        metrics[f"vonmises_{sf_idx}_r2_score"] = r2

            # Normalized responses for each direction
            norm_resp = []
            mean_grating_response = roi_trial_resp.mean(skipna=True).item()
            mean_blank_response = self.blank_responses[roi].mean()
            for i in range(len(self.dir_list)):
                resp = roi_trial_resp.isel(direction=i, spatial_frequency=pref_sf_idx).mean(skipna=True).item()
                norm_resp.append((resp - mean_blank_response) / (mean_grating_response + mean_blank_response))
            metrics["norm_dir_responses"] = norm_resp


        metrics = pd.DataFrame(data=all_metrics)

        metrics["is_valid"] = self.is_roi_valid # ~metrics.isna().any(axis=1)

        if "pref_dir" in metrics.columns:
            metrics["pref_ori"] = np.mod(metrics.pref_dir, 180)
        # metrics["pref_ori_naive"] = np.mod(metrics.pref_dir_naive, 180)


        # Is responsive using chi-squared test
        metrics["chisq_response_p"] = get_chisq_response_proba(self.stim_table, ["direction", "spatial_frequency"], self.sweep_responses, n_shuffles=self.n_chisq_shuffles)

        # Null distribution
        metrics["null_dist_multi_mean"] = null_multi_mean
        metrics["null_dist_multi_std"] = null_multi_std
        metrics["null_dist_single_mean"] = self.null_dist_single_trial.mean(axis=1)
        metrics["null_dist_single_std"] = self.null_dist_single_trial.std(axis=1)

        metrics = metrics.convert_dtypes()
        self._metrics = metrics
        # print(" Done.")

    @staticmethod
    def get_orth_dir_indices(dir_i: int):
        # Returns two indices in dir_list for the two relative 90-degree orthogonal directions
        orth_1_idx = (dir_i + 3) % 12
        orth_2_idx = (dir_i - 3) % 12
        return orth_1_idx, orth_2_idx
    
    @staticmethod
    def get_null_dir_index(dir_i: int):
        # Returns index in dir_list for the relative 180-degree opposite direction
        return (dir_i + 6) % 12

    # Plotting utils

    





    def plot_roi_stim_conditions(self, roi: int, plot_time_window: tuple=(-0.5, 2.5), plot_individual_traces: bool=False, use_baseline_normalized_traces: bool=True, n_baseline_samples=10000, trace_type: str="dff"):
        roi_traces = self.get_traces(trace_type).sel(roi=roi)
        frame_dur = np.mean(np.diff(roi_traces.time))
        window_start = int(np.floor(plot_time_window[0] / frame_dur))
        window_end = int(np.ceil(plot_time_window[1] / frame_dur))
        time_axis = frame_dur * np.arange(window_start, window_end)
        
        # Compute a null distribution from spontaneous stimulus
        if trace_type == "dff":
            baseline_frame_window = (-3, 0) if use_baseline_normalized_traces else None
            baseline_dff = self.get_spont_null_dist_dff_traces(roi, (window_start, window_end), baseline_frame_window=baseline_frame_window, n_boot=n_baseline_samples)
            baseline_low, baseline_high = np.quantile(baseline_dff, [0.025, 0.975], axis=0) # middle 95% of baseline

        fig = plt.figure(figsize=(12, 5))
        subplots = []
        ylim = [None, None]
        mean_responses = []

        for dir_i, dir in enumerate(self.dir_list):
            for sf_i, sf in enumerate(self.sf_list):
                ax = fig.add_subplot(len(self.sf_list), len(self.dir_list), sf_i*len(self.dir_list)+dir_i+1)
                subplots.append(ax)

                # Find a set of stimulus traces
                stim_idx = self.get_stim_idx(dir, sf)
                stim_traces = np.empty((len(stim_idx), window_end-window_start), dtype=float)

                for trial_i, stim_i in enumerate(stim_idx):
                    time_idx = roi_traces.indexes["time"].get_loc(self.stim_table.at[stim_i, "start"], method="nearest")
                    trace_slice = slice(time_idx+window_start, time_idx+window_end)
                    trace = roi_traces.isel(time=trace_slice)
                    
                    if trace_type == "dff" and use_baseline_normalized_traces:
                        base_slice = slice(time_idx+baseline_frame_window[0], time_idx+baseline_frame_window[1])
                        trace = trace - roi_traces.isel(time=base_slice).mean()
                    
                    stim_traces[trial_i] = trace

                # Save mean trial response (to be used in background color later)
                mean_responses.append(self.trial_responses.sel(roi=roi, direction=dir, spatial_frequency=sf).mean(skipna=True).item())

                if trace_type == "dff":
                    # Plot the mean trace for this stimulus condition
                    mean_trace = stim_traces.mean(axis=0)
                    ax.plot(time_axis, mean_trace, color="black")

                    if plot_individual_traces:
                        ax.plot(time_axis, stim_traces.T, color="black", lw=0.2)
                    
                    # Plot a null distribution
                    ax.fill_between(time_axis, baseline_low, baseline_high, alpha=0.6, color="darkgray") # plot null distribution
                    
                    # Update ticks
                    ax.set_xticks([0, 2]) # window of the actual stimulus
                    plt.setp(ax.get_xticklabels(), fontsize=10)
                    plt.setp(ax.get_yticklabels(), fontsize=12)
                    
                    # So all subplots have the same y-axis limits
                    ax_ylim = ax.get_ylim()
                    if ylim[0] is None or ax_ylim[0] < ylim[0]:
                        ylim[0] = ax_ylim[0]
                    if ylim[1] is None or ax_ylim[1] > ylim[1]:
                        ylim[1] = ax_ylim[1]
                    # if ylim[0] is None or np.min(mean_trace) < ylim[0]:
                    #     ylim[0] = np.min(mean_trace)
                    # if ylim[1] is None or np.max(mean_trace) > ylim[1]:
                    #     ylim[1] = np.max(mean_trace)

                    # y-axis labels on left subplots
                    if dir_i == 0:
                        ax.set_ylabel(f"SF = {sf:.2f} cpd", fontsize=16)
                    else:
                        ax.get_yaxis().set_visible(False)

                    # x-axis labels on bottom subplots
                    if sf_i == len(self.sf_list)-1:
                        ax.set_xlabel(f"{int(dir)}°", fontsize=16)
                    else:
                        ax.get_xaxis().set_visible(False)
                elif trace_type == "events":
                    for i in range(stim_traces.shape[0]):
                        for j in range(stim_traces.shape[1]):
                            e = stim_traces[i, j]
                            if e > 0:
                                t = time_axis[j]
                                ax.plot([i, i+0.8], [t, t], color="black", linewidth=max(0.5, e*20))
                    
                    ax.set_xlim(-0.1, stim_traces.shape[0]+0.1)
                    ax.set_ylim(time_axis.max(), time_axis.min())
                    ax.axhspan(ymin=0, ymax=2, color="gray", alpha=0.3)
                    # ax.axhline(y=0, color="black", linestyle="dashed", linewidth=1)
                    # ax.axhline(y=2, color="black", linestyle="dashed", linewidth=1)
                    # ax.set_title(len(stim_idx), fontsize=10)
                    ax.set_xticks([])

                    # y-axis labels on left subplots
                    if dir_i == 0:
                        ax.set_ylabel(f"SF = {sf:.2f} cpd", fontsize=16)
                    else:
                        ax.get_yaxis().set_visible(False)

                    # x-axis labels on bottom subplots
                    if sf_i == len(self.sf_list)-1:
                        ax.set_xlabel(f"{int(dir)}°", fontsize=16)

        # Add colors and scale y-axis
        # vpad = 0.05*(ylim[1]-ylim[0])
        # ylim[0] -= vpad
        # ylim[1] += vpad
        cmap = plt.get_cmap("RdBu_r")
        max_response_abs = max(abs(r) for r in mean_responses)
        cmap_norm = mpl.colors.Normalize(vmin=-max_response_abs, vmax=max_response_abs)

        for ax, mean_response in zip(subplots, mean_responses):
            ax.set_facecolor(cmap(cmap_norm(mean_response)))

            if trace_type == "dff":
                ax.set_xlim(plot_time_window)
                ax.set_ylim(ylim)
                ax.axvline(x=0, color="black", linestyle="dashed", linewidth=1)
                ax.axvline(x=2, color="black", linestyle="dashed", linewidth=1)
                # ax.axvspan(0, 2, alpha=0.4, color="#777777") # Shade the response window in gray
                ax.axhline(y=0, color="black", linestyle="dashed", linewidth=1)

        frac_resp_trials = self.metrics.at[roi, "frac_responsive_trials"]
        pref_dir = self.metrics.at[roi, "pref_dir"]
        pref_sf = self.metrics.at[roi, "pref_sf"]
        title = [
            f"Mean DG-{self.dg_type.upper()} norm. {trace_type} traces for ROI {roi} in plane {self.plane} of session {self.session.get_session_id()}",
            f"Frac Resp Trials={frac_resp_trials:.2f}, Pref Dir={pref_dir}°, Pref SF={pref_sf:.2f} cpd, OSI={self.metrics.at[roi, 'osi']:.2f}, DSI={self.metrics.at[roi, 'dsi']:.2f}"
        ]
        fig.suptitle("\n".join(title), fontsize=14)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.25)
        # return fig



    def plot_roi_tuning_curves(self, roi: int, dg_other=None, plot_all_sf=False, figsize=(8, 4)):
        if dg_other is not None:
            dg_full = (self if self.dg_type == "full" else dg_other)
            dg_windowed = (self if self.dg_type == "windowed" else dg_other)
            if dg_full is None or dg_windowed is None:
                raise ValueError(f"if dg_other is given it must be different dg_type from self (both are DG-{self.dg_type})")

        def plot_dir_tuning(dgs, ax, color="black"):
            sf_colors = ["r", "b"]

            for sf_idx in range(len(self.sf_list)):
                for dg in dgs:
                    resp_matrix = dg.trial_responses.sel(roi=roi).isel(spatial_frequency=sf_idx)
                    y = resp_matrix.mean("trial", skipna=True)
                    num_trials = resp_matrix.count("trial")
                    y_err = resp_matrix.std("trial", skipna=True) / np.sqrt(num_trials) # SEM error
                    label = f"{dg.dg_type.capitalize()} (at {dg.sf_list[sf_idx]:.2f} cpd)"
                    ax.errorbar(dg.dir_list, y, y_err, marker="o", label=label, color=sf_colors[sf_idx], linewidth=(4 if dg.dg_type == "full" else 1))

            # pref_sf_idx = dg.metrics.at[roi, "pref_sf_idx"]
            # if plot_all_sf:
            #     sf_idx_to_plot = list(range(len(dg.sf_list)))
            # else:
            #     sf_idx_to_plot = [pref_sf_idx]
            # for sf_idx in sf_idx_to_plot:
            #     resp_matrix = dg.trial_responses.sel(roi).isel(spatial_frequency=sf_idx) # shape (dir, trials)
            #     y = np.nanmean(resp_matrix, axis=1) # shape (dir,)
            #     num_trials = np.count_nonzero(~np.isnan(resp_matrix), axis=1) # shape (dir,)
            #     y_err = np.nanstd(resp_matrix, axis=1) / np.sqrt(num_trials) # SEM error
            #     is_pref = sf_idx == pref_sf_idx
            #     labelpref = ", pref" if is_pref else ""
            #     label = f"{dg.dg_type.capitalize()} (at {dg.sf_list[sf_idx]:.2f} cpd{labelpref})"
            #     linestyle = "solid" if is_pref else "dotted"
            #     alpha = 1 if is_pref else 0.5
            #     ax.errorbar(dg.dir_list, y, y_err if is_pref else None, linestyle=linestyle, marker="o", color=color, alpha=alpha, label=label)

        fig, (ax_heatmap, ax_dir) = plt.subplots(ncols=2, figsize=figsize, gridspec_kw=dict(width_ratios=[1, 5]))

        # from matplotlib.patches import Rectangle

        # Plot heatmap
        if dg_other is None:
            mean_resp_matrix = np.nanmean(self.trial_responses[roi], axis=2) # trial average, shape [dir x SF]
            max_resp_abs = np.max(np.abs(mean_resp_matrix))
            cmap = plt.get_cmap("RdBu_r")
            cmap_norm = mpl.colors.Normalize(vmin=-max_resp_abs, vmax=max_resp_abs) # linear color normalization
            ax_heatmap.pcolormesh(mean_resp_matrix, cmap=cmap, norm=cmap_norm)
            ax_heatmap.set_xticks(ticks=[x+0.5 for x in range(len(self.sf_list))], labels=[f"{sf:.2f}" for sf in self.sf_list])
        else:
            resp_full = np.nanmean(dg_full.trial_responses[roi], axis=2)
            resp_windowed = np.nanmean(dg_windowed.trial_responses[roi], axis=2)
            mean_resp_matrix = np.column_stack([resp_full[:, 0], resp_windowed[:, 0], resp_full[:, 1], resp_windowed[:, 1]])
            max_resp_abs = np.max(np.abs(mean_resp_matrix))
            cmap = plt.get_cmap("RdBu_r")
            cmap_norm = mpl.colors.Normalize(vmin=-max_resp_abs, vmax=max_resp_abs) # linear color normalization
            ax_heatmap.pcolormesh(mean_resp_matrix, cmap=cmap, norm=cmap_norm)

            xticks = []
            xticklabels = []

            for i, sf in enumerate(self.sf_list):
                xticks.append(2*i+0.5)
                xticks.append(2*i+1.5)
                sf = f"{sf:.2f}".lstrip("0")
                xticklabels.append(f"{sf}\nF")
                xticklabels.append(f"{sf}\nW")

            ax_heatmap.set_xticks(ticks=xticks, labels=xticklabels)

            
            # box the largest responses
            # ax_heatmap.add_patch(Rectangle((0, height), 1, -1, linewidth=1, edgecolor="blue", facecolor="none", zorder=10))
            

            # height = len(self.dir_list)
            # ax_heatmap.add_patch(Rectangle((0, height), 1, -height, linewidth=1, edgecolor="blue", facecolor="none", zorder=10))
            # ax_heatmap.add_patch(Rectangle((1, height), 1, -height, linewidth=1, edgecolor="red", facecolor="none", zorder=10))
            # ax_heatmap.add_patch(Rectangle((2, height), 1, -height, linewidth=1, edgecolor="blue", facecolor="none", zorder=10))
            # ax_heatmap.add_patch(Rectangle((3, height), 1, -height, linewidth=1, edgecolor="red", facecolor="none", zorder=10))


        ax_heatmap.set_yticks(ticks=[x+0.5 for x in range(len(self.dir_list))], labels=[f"{dir:.0f}" for dir in self.dir_list])
        ax_heatmap.set_xlabel("Spatial frequency (cpd)", fontsize=12)
        ax_heatmap.set_ylabel("Direction (°)", fontsize=12)

        if dg_other is None:
            plot_dir_tuning([self], ax_dir)
        else:
            plot_dir_tuning([dg_windowed, dg_full], ax_dir)

        ax_dir.axhline(y=np.quantile(self.null_dist_multi_trial[roi], 0.95), label="Baseline 95%", color="black", linestyle="dashed")

        ax_dir.set_xticks(self.dir_list)
        ax_dir.tick_params(axis="both", labelsize=12)
        ax_dir.set_xlabel(f"Direction (°)", fontsize=12)
        ax_dir.set_ylabel(f"Mean response ({self.trace_type})", fontsize=12)
        ax_dir.legend(fontsize=10)        


        title = f"DG tuning curves for ROI {roi} in plane {self.plane} of session {self.session.get_session_id()}"
        metrics_title = "" if self._metrics is None else f"\nosi={self.metrics.at[roi, 'osi']:.2f}, gosi={self.metrics.at[roi, 'gosi']:.2f}, dsi={self.metrics.at[roi, 'dsi']:.2f}"
        title = f"{title}{metrics_title}"
        fig.suptitle(title, fontsize=14)

        fig.tight_layout()
        # return fig


    def plot_trial_responses(self, roi: int, response_type="raw", ax=None):
        # response_type can be "raw" (mean event) or "z_score" (z-score compared to null dist)
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3.5))
        pref_sf_idx = self.metrics.at[roi, "pref_sf_idx"]
        pref_dir_idx = self.metrics.at[roi, "pref_dir_idx"]

        null_mean = self._null_dist_single_trial[roi].mean()
        null_std = self._null_dist_single_trial[roi].std()

        for sf_i, sf in enumerate(self.sf_list):
            color = "purple" if sf_i == pref_sf_idx else "darkgray"
            
            for dir_i, dir in enumerate(self.dir_list):
                responses = self.trial_responses[roi, dir_i, sf_i]
                responses = responses[~np.isnan(responses)]

                x, y = [], []

                for response in responses:
                    x.append(dir)

                    if response_type == "raw":
                        y.append(response)
                    elif response_type == "z_score":
                        y.append((response - null_mean) / null_std)
                
                alpha = 0.5

                if sf_i == pref_sf_idx:
                    if dir_i == pref_dir_idx:
                        color = "red"
                        alpha = 1
                    else:
                        color = "purple"
                else:
                    color == "gray"

                ax.scatter(x, y, color=color, alpha=alpha, label=f"{sf:.2} cpd", marker="x")

                # Plot mean response
                mean_resp = np.mean(responses)
                ax.hlines(y=mean_resp, xmin=dir-5, xmax=dir+5, color=color, linestyle="solid", linewidth=3)
            
        ax.legend(handles=[
            mpl.lines.Line2D([0], [0], marker="X", color="w", markersize=10, markerfacecolor="red", label=f"Pref: {self.sf_list[pref_sf_idx]:.2f} cpd, {self.dir_list[pref_dir_idx]:.0f}°"),
            mpl.lines.Line2D([0], [0], marker="X", color="w", markersize=10, markerfacecolor="purple", label="Pref TF, Not Pref Dir"),
            mpl.lines.Line2D([0], [0], marker="X", color="w", markersize=10, markerfacecolor="gray", label="Not Pref TF"),
            mpl.lines.Line2D([0], [0], color="black", linewidth=3, label="Condition Mean"),
            mpl.lines.Line2D([0], [0], color="gray", linestyle="dashed", label="Null Dist 95%ile"),
            # mpl.patches.Patch(facecolor="gray", label="Null Dist 95%")
        ], bbox_to_anchor=(1, 1), loc="upper left")
        ax.set_title(f"DG-{self.dg_type} trial responses for ROI {roi} in plane {self.plane} of session {self.session.get_session_id()}")
        ax.set_xticks(self.dir_list)
        ax.set_xlabel("Direction (°)")
        if response_type == "raw":
            ax.axhline(y=np.quantile(self._null_dist_single_trial[roi], 0.95), color="gray", linestyle="dashed")
            ax.set_ylabel("Mean event response to trial")
        elif response_type == "z_score":
            ax.axhline(y=(np.quantile(self._null_dist_single_trial[roi], 0.95) - null_mean) / null_std, color="gray", linestyle="dashed")
            ax.set_ylabel("Trial response Z-score")

        # fig.tight_layout()
        return ax


    def plot_waterfall_normalized_responses(self, figsize=(15, 5), included_rois=None, center_zero_direction=False):
        if included_rois is None:
            included_rois = self.metrics.index[self.is_roi_valid & (self.metrics.frac_responsive_trials >= 0.5)]

        # inclusion = metrics.is_responsive
        # inclusion = metrics.frac_responsive_trials >= 0.5
        n_rois = len(included_rois)
        dir_list = self.dir_list
        sf_list = self.sf_list
        n_dir = len(self.dir_list)
        n_sf = len(self.sf_list)
        
        dir_idx = np.arange(len(dir_list))
        if center_zero_direction:
            dir_idx = np.mod(dir_idx + len(dir_idx)//2 + 1, len(dir_idx)) # second half before first half
            # dir_list = dir_list[dir_idx]
            dir_list = np.where(dir_list > 180, dir_list-360, dir_list)

        matrix = np.zeros((n_rois, n_dir*n_sf))
        mean_trial_responses = self.trial_responses.sel(roi=included_rois).mean("trial", skipna=True)
        mean_grating_response = mean_trial_responses.mean(dim=("direction", "spatial_frequency"), skipna=True).values.reshape(n_rois, 1) # mean response across all trials; (n_rois, 1)
        mean_blank_response = self.blank_responses[included_rois].mean(axis=1).reshape(n_rois, 1) # mean response to blank screen; (n_rois, 1)

        for sf_i in range(n_sf):
            mean_resp = mean_trial_responses.isel(direction=dir_idx, spatial_frequency=sf_i) # (n_rois, n_dir)
            norm_resp = (mean_resp - mean_blank_response) / (mean_grating_response + mean_blank_response)
            matrix_j = sf_i*n_dir
            matrix[:, matrix_j:matrix_j+n_dir] = norm_resp

        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        sort_idx = np.argsort(matrix.argmax(axis=1) % n_dir)
        sort_idx = np.argsort(dir_idx[self.metrics.pref_dir_idx[included_rois].astype(int)])
        c = ax.pcolormesh(matrix[sort_idx], cmap="RdBu_r", vmin=-3, vmax=3)
        cbar = fig.colorbar(c, ax=ax, pad=.01)
        cbar.set_label("Normalized mean response")
        for i in range(1, n_sf):
            ax.axvline(x=i*n_dir, color="black")

        

        # y-axis: ROI
        ax.invert_yaxis()
        ax.set_ylabel("ROI", fontsize=12)
        ax.set_yticks([n_rois])

        # x-axis, bottom: Spatial frequency
        ax.set_xticks(np.arange(n_sf)*n_dir + n_dir/2)
        ax.set_xticklabels([f"{sf:.2f} cpd" for sf in sf_list], fontsize=12)

        # x-axis, top: Direction
        xaxis_top = ax.twiny()
        xaxis_top.set_xlim(ax.get_xlim())
        dir_ticklabels = [f"{dir_list[i]}°" if i % 3 == 0 else "" for i in dir_idx] * n_sf
        xaxis_top.set_xticks(np.arange(len(dir_ticklabels))+0.5)
        xaxis_top.set_xticklabels(dir_ticklabels)

        return ax


    def get_mean_response(self, roi: int, dir_idx: int=None, sf_idx: int=None, return_error: bool=False) -> float:
        """Get the mean response to a given stimulus condition.
        If dir_idx or sf_idx is null, it defaults to the preferred condition.

        Args:
            roi (int): ROI
            dir_idx (int, optional): Index of direction. Defaults to None (i.e., preferred direction).
            sf_idx (int, optional): Index of spatial frequency. Defaults to None (i.e., preferred spatial frequency).
            return_std (bool, optional): Whether to return a tuple of (mean, err), where err is the SEM. Defaults to False.

        Returns:
            float: Mean trial response for the ROI to the given stimulus condition.
        """
        if dir_idx is None:
            dir_idx = self.metrics.loc[roi, "pref_dir_idx"]
        if sf_idx is None:
            sf_idx = self.metrics.loc[roi, "pref_sf_idx"]
        
        if pd.isna(dir_idx) or pd.isna(dir_idx):
            return (np.nan, np.nan) if return_error else np.nan

        responses = self.trial_responses.sel(roi=roi).isel(direction=dir_idx, spatial_frequency=sf_idx).dropna("trial")
        mean = responses.mean().item()
        
        if return_error:
            return mean, responses.std().item() / np.sqrt(len(responses))
        else:
            return mean

    # TODO: need to clean up these next two SS computation methods

    @staticmethod
    def compute_ss_metrics_single_plane(dgf, dgw) -> pd.DataFrame:
        columns = {
            "dgw_resp_pref_dgf": ([], float),
            "dgf_resp_pref_dgw": ([], float),
            "ssi": ([], float),
            "ffsi": ([], float),
            "ssi_pref_both": ([], float),
            "ssi_orth": ([], float),
            "ffsi_all_dirs": ([], object),
        }

        def append_row(**kwargs):
            for column in columns:
                val = kwargs.get(column, None)
                columns[column][0].append(val)

        def ratio(a, b):
            return 0 if b == 0 else a/b

        for roi in dgf.session.get_rois(dgf.plane):
            dir_w = dgw.metrics.loc[roi, "pref_dir_idx"]
            sf_w = dgw.metrics.loc[roi, "pref_sf_idx"]
            dir_f = dgf.metrics.loc[roi, "pref_dir_idx"]
            sf_f = dgf.metrics.loc[roi, "pref_sf_idx"]
            
            resp_w = dgw.get_mean_response(roi, dir_w, sf_w)
            resp_f = dgf.get_mean_response(roi, dir_f, sf_f)
            resp_w_pref_f = dgw.get_mean_response(roi, dir_f, sf_f)
            resp_f_pref_w = dgf.get_mean_response(roi, dir_w, sf_w)

            is_nan = np.isnan(resp_w) or np.isnan(resp_f)
            ssi = np.nan if is_nan else ratio(resp_w - resp_f_pref_w, resp_w + resp_f_pref_w)
            ffsi = np.nan if is_nan else ratio(resp_w - resp_f_pref_w, resp_w)
            ssi_pref_both = np.nan if is_nan else ratio(resp_w - resp_f, resp_w + resp_f)
            ffsi_all_dirs = []

            if is_nan:
                ssi_orth = np.nan
            else:
                orth_1, orth_2 = dgw.get_orth_dir_indices(dir_w)
                orth_1_w = dgw.get_mean_response(roi, orth_1, sf_w)
                orth_2_w = dgw.get_mean_response(roi, orth_2, sf_w)
                orth_1_f = dgf.get_mean_response(roi, orth_1, sf_w)
                orth_2_f = dgf.get_mean_response(roi, orth_2, sf_w)

                ssi_orth_1 = ratio(orth_1_w - orth_1_f, orth_1_w + orth_1_f)
                ssi_orth_2 = ratio(orth_2_w - orth_2_f, orth_2_w + orth_2_f)
                ssi_orth = (ssi_orth_1 + ssi_orth_2) / 2

                for dir_i in range(len(dgf.dir_list)):
                    # print(roi, dir_i, type(dir_i), sf_w, type(sf_w))
                    rw = dgw.get_mean_response(roi, dir_i, sf_w)
                    rf = dgf.get_mean_response(roi, dir_i, sf_w)
                    ffsi_dir = ratio(rw - rf, rw)
                    ffsi_all_dirs.append(ffsi_dir)

            # Same index as in StimulusAnalysis.concat_metrics
            append_row(
                dgw_resp_pref_dgf=resp_w_pref_f,
                dgf_resp_pref_dgw=resp_f_pref_w,
                ssi=ssi,
                ffsi=ffsi,
                ssi_pref_both=ssi_pref_both,
                ssi_orth=ssi_orth,
                ffsi_all_dirs=ffsi_all_dirs,
            )
            
            # ss_metrics.loc[index] = (int(mouse), int(column), volume, plane, roi, ss)
        
        ss_metrics = pd.DataFrame(data={
            column: np.array(values, dtype=dtype)
            for column, (values, dtype) in columns.items()
        })

        return ss_metrics


    @staticmethod
    def compute_surround_suppression_metrics(dg_full, dg_windowed) -> pd.DataFrame:
        """Compute the surround suppression metrics for a set of DG-full and DG-windowed analyses.

        Args:
            dg_full (DriftingGratings or list[DriftingGratings]): DG analysis objects for full-field DG stimuli.
            dg_windowed (DriftingGratings or list[DriftingGratings]): DG analysis objects for windowed DG stimuli.

        Returns:
            pd.DataFrame: Surround suppression metrics for each ROI (across potentially many mice/columns/volumes/planes, depending on inputs).
        """
        reindex = True
        # Make arrays if neccessary
        if type(dg_full) not in (list, tuple):
            dg_full = [dg_full]
            reindex = False
        if type(dg_windowed) not in (list, tuple):
            dg_windowed = [dg_windowed]
            reindex = False

        all_ss_metrics = []

        for dgf, dgw in zip(dg_full, dg_windowed):
            assert dgf.plane == dgw.plane
            mouse, column, volume = dgf.session.get_mouse_column_volume()
            plane = dgf.plane
            ss_metrics = DriftingGratings.compute_ss_metrics_single_plane(dgf, dgw)
            ss_metrics.insert(0, "mouse", mouse)
            ss_metrics.insert(1, "column", column)
            ss_metrics.insert(2, "volume", volume)
            ss_metrics.insert(3, "plane", plane)
            ss_metrics.insert(4, "roi", ss_metrics.index)
            ss_metrics.insert(5, "is_valid", dgf.is_roi_valid)
            ss_metrics.insert(6, "depth", dgf.session.get_plane_depth(plane))

            if reindex:
                ss_metrics.index = [f"M{mouse}_{column}{volume}_{plane}_{roi}" for roi in ss_metrics.index]
            
            all_ss_metrics.append(ss_metrics)
        
        if len(all_ss_metrics) == 0:
            return None
        else:
            return pd.concat(all_ss_metrics, axis=0)
