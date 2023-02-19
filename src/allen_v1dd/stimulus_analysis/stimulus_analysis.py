import numpy as np
import pandas as pd
import xarray as xr

class StimulusAnalysis(object):
    """Generic class used for analyzing neural responses to stimuli.
    Designed to be subclassed to analyze particular stimuli.
    """

    TIME_PER_FRAME = 0.17 # rounded up so time windows include frames

    def __init__(self, stim_name: str, stim_abbrev: str, session, plane: int, trace_type: str):
        self.stim_name = stim_name
        self.stim_abbrev = stim_abbrev
        self.session = session
        self.plane = plane
        self.trace_type = trace_type

        self.stim_table, self.stim_meta = session.get_stimulus_table(stim_name)
        self.is_roi_valid = session.is_roi_valid(plane)
        self.n_rois = len(self.is_roi_valid)
        self.n_rois_valid = np.count_nonzero(self.is_roi_valid)
        self._null_dist_cache = {}
    
    @property
    def spont_stim_table(self):
        return self.session.get_stimulus_table("spontaneous")[0]

    def get_traces(self, trace_type: str = None):
        """Shorthand to get traces for this plane.

        Args:
            trace_type (str, optional): Type of trace. Defaults to None, which is self.trace_type.

        Returns:
            array_like: Traces for this plane.
        """
        if trace_type is None:
            trace_type = self.trace_type
        return self.session.get_traces(plane=self.plane, trace_type=trace_type)

    def get_responses(self, time: float, baseline_time_window: tuple, response_time_window: tuple, trace_type: str=None) -> xr.DataArray:
        """Compute the response of each ROI at a given frame. Response is defined as
        (mean trace during response_time_window) - (mean traceduring baseline_time_window, or 0 if baseline_time_window is None)
        If baseline_time_window is None, then only the trace during response_frame_window is used (i.e., no normalization).

        Args:
            time (float): Time of response start.
            baseline_time_window (tuple): Time window to use for computing baseline.
            response_time_window (tuple): Same as baseline_time_window, but used for computing response.
            trace_type (str, optional): Type of trace. Defaults to None, which is self.trace_type.

        Returns:
            array_like: 1d array of size n_rois containing the response, as defined above, of each ROI.
        """
        traces = self.get_traces(trace_type)
        # start = traces.indexes["time"].get_loc(time+response_time_window[0], method="nearest")
        # end = traces.indexes["time"].get_loc(time+response_time_window[1], method="nearest")
        # response = traces.isel(time=(start, end))
        response = traces.sel(time=slice(time+response_time_window[0], time+response_time_window[1])).mean("time")

        if baseline_time_window is None:
            return response
        else:
            baseline = traces.sel(time=slice(time+baseline_time_window[0], time+baseline_time_window[1])).mean("time")
            return response - baseline

    def get_random_spont_times(self, shape, start_padding=2, end_padding=-2):
        start, end = self.spont_stim_table.at[0, "start"], self.spont_stim_table.at[0, "end"]
        random_times = np.random.uniform(low=start+start_padding, high=end+end_padding, size=shape)
        return random_times

    def get_spont_null_dist_dff_traces(self, roi, frame_window, baseline_frame_window=None, n_boot=1000):
        # Used to show a baseline distribution when plotting for an individual ROI
        roi_dff = self.get_traces("dff").sel(roi=roi)
        # time_used_before = max(-frame_window[0], 0 if baseline_frame_window is None else -baseline_frame_window[0])
        
        trace_len = frame_window[1] - frame_window[0]
        random_times = self.get_random_spont_times(shape=n_boot)
        dist = np.empty((n_boot, trace_len), dtype=float)

        for boot_i, time in enumerate(random_times):
            time_idx = roi_dff.indexes["time"].get_loc(time, method="nearest")
            time_slice = slice(time_idx+frame_window[0], time_idx+frame_window[1])
            trace = roi_dff.isel(time=time_slice)

            if baseline_frame_window is not None:
                baseline_slice = slice(time_idx+baseline_frame_window[0], time_idx+baseline_frame_window[1])
                trace = trace - roi_dff.isel(time=baseline_slice).mean()
            
            dist[boot_i, :] = trace
        
        return dist

    def get_spont_null_dist(self, baseline_time_window: tuple, response_time_window: tuple, n_boot: int=1000, n_means: int=1, trace_type: str="events", cache: bool=True) -> np.ndarray:
        """Returns a bootstrap distribution of randomly sampled (with replacement) responses during the spontaneous stimulus.

        Args:
            baseline_time_window (tuple): Time window used to compute baseline.
            response_time_window (tuple): Same as baseline_time_window, but used to compute response.
            n_boot (int, optional): Number of bootstrap random samples. Defaults to 1000.
            n_means (int, optional): Number of responses that are averaged for each sample in the null distribution. Defaults to 1.
            trace_type (str, optional): Type of trace to use for computing the null dist; either "events" or "dff". Defaults to "events".
            cache (bool, optional): Whether to use a cache lookup to avoid expensive computation. Defaults to True.
        
        Returns:
            np.ndarray: Bootstrap distribution of spontaneous responses; has shape (n_ROI, n_boot) = (traces.shape[0], n_boot).
        """
        # I believe the timestamps are the same but just doing this for clarity sake
        cache_key = (baseline_time_window, response_time_window, n_boot, n_means, trace_type)

        if cache_key in self._null_dist_cache:
            return self._null_dist_cache[cache_key]

        random_times = self.get_random_spont_times(
            shape=(n_boot, n_means),
            start_padding=(-baseline_time_window[0] if baseline_time_window is not None else 0),
            end_padding=response_time_window[1]
        )
        dist = np.empty((self.n_rois, n_boot))

        for boot_i in range(n_boot):
            r = 0
            rand_times = random_times[boot_i]
            for time in rand_times:
                r += self.get_responses(time=time, baseline_time_window=baseline_time_window, response_time_window=response_time_window, trace_type=trace_type)
            dist[:, boot_i] = r / n_means
        
        self._null_dist_cache[cache_key] = dist
        return dist


    @staticmethod
    def concat_metrics(analyses):
        all_metrics = []
        stim_name = None

        for analysis in analyses:
            # Validate that all analyses are for the same stimuli
            name = analysis.stim_name

            if stim_name is None:
                stim_name = name
            elif stim_name != name:
                raise ValueError(f"all analyses must be for same stimulus, but given analyses for {stim_name} and {name}")

            # Add columns for plane and ROI
            metrics = analysis.metrics.copy()
            mouse, column, volume = analysis.session.get_mouse_column_volume()
            metrics.insert(0, "mouse", mouse)
            metrics.insert(1, "column", column)
            metrics.insert(2, "volume", volume)
            metrics.insert(3, "plane", analysis.plane)
            metrics.insert(4, "roi", metrics.index)
            metrics.insert(5, "depth", analysis.session.get_plane_depth(analysis.plane))
            metrics.index = [f"M{mouse}_{column}{volume}_{analysis.plane}_{roi}" for roi in metrics.index]
            
            # Merge the metrics
            all_metrics.append(metrics)
        
        if len(all_metrics) == 0:
            return None
        else:
            return pd.concat(all_metrics)