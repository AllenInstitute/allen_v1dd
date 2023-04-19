from os import path
from glob import glob
from collections.abc import Iterable

import numpy as np
import h5py
import pandas as pd
import xarray as xr

# from ..stimulus_analysis.timing_utils import find_nearest

class OPhysSession:
    """Represents a ophys session data wrapper."""

    def __init__(self, v1dd_client, session_id: str, nwb_file_path: str):
        """Initialize a new ophys session data wrapper.

        Args:
            v1dd_client (V1DDClient): V1DD client
            session_id (str): Session ID
            nwb_file_path (str): NWB file path
        """
        
        self.v1dd_client = v1dd_client
        self.session_id = session_id
        self.nwb_file_path = nwb_file_path
        self._stim_table_cache = {}

        with h5py.File(nwb_file_path, "r") as nwb_file:
            self._check_integrity(nwb_file)

            self.lims_session_id = nwb_file["general/session_id"][()].decode() # Decode the binary string
            self.stim_list = list(nwb_file["stimulus/presentation"].keys())

            device = nwb_file["general/optophysiology/imaging_plane_0/device"][()].decode()
        
            if device == "two-photon scope":
                device = "2p"
            elif device == "three-photon scope":
                device = "3p"
            else:
                device = f"Unknown device: {device}"
            
            self.scope_type = device

            # Load planes
            self._planes = []
            self._plane_rois = {}
            self._plane_depths = {}
            self._trace_cache = {}
            self._plane_pika_roi_conf = {}

            for key in nwb_file["processing"].keys():
                if key.startswith("rois_and_traces_plane"):
                    plane_name = key[key.rindex("_")+1:]
                    plane = int(plane_name[5:]) # Remove the "plane"
                    self._planes.append(plane)

                    rois = nwb_file[f"{self._rois_and_traces(plane)}/ImageSegmentation/imaging_plane"].keys()
                    rois = [int(r.split("_")[1]) for r in rois if r.startswith("roi_") and not r.endswith("_list")]
                    rois.sort()
                    self._plane_rois[plane] = rois
                    self._plane_depths[plane] = nwb_file[f"{self._rois_and_traces(plane)}/imaging_depth_micron"][()]
                    self._plane_pika_roi_conf[plane] = nwb_file[f"analysis/roi_classification_pika/plane{plane}/score"][()]


        # Parse mouse, column, and volume
        # session_id = "M......_.."
        #               0123456789
        def int_or_str(x):
            try:
                return int(x)
            except ValueError:
                return str(x)
        self.mouse_id = int(session_id[1:7])
        self.column_id = int(session_id[8])
        self.volume_id = int_or_str(session_id[9])

    def open_file(self):
        return h5py.File(self.nwb_file_path, "r")

    def get_stimulus_table(self, stim_name: str) -> pd.DataFrame:
        # Check cache
        if stim_name in self._stim_table_cache:
            return self._stim_table_cache[stim_name]

        stim_table, stim_meta = None, None

        with self.open_file() as nwb_file:
            nwb_group = nwb_file["stimulus/presentation"]

            if stim_name not in nwb_group:
                raise ValueError(f"unable to get stimulus table for {stim_name}")
            
            nwb_group = nwb_group[stim_name]
            data = nwb_group["data"]

            def get_meta(key, default=None):
                if default is None or key in nwb_group:
                    return nwb_group[key][()]
                else:
                    return default
                
            if stim_name == "drifting_gratings_full" or stim_name == "drifting_gratings_windowed":
                dg_type = stim_name[stim_name.rindex("_")+1:]
                stim_table = pd.DataFrame(data=data, columns=["start", "end", "temporal_frequency", "spatial_frequency", "direction"]) # last column is internally labeled "orientation" but it is actually a direction
                stim_meta = {
                    "temporal_frequency": get_meta("TF_Hz"),
                    "contrast": get_meta("contrast"),
                    "duration": get_meta("duration_sec"),
                    "padding": 1.0, # time in between successive stimuli
                    "num_samples": get_meta("num_samples"),
                    "n_trials": get_meta("iteartion"), # NOT A TYPO -- this is an unfortnuate typo in the NWB files ;)
                    "center_position": (get_meta("center_alt"), get_meta("center_azi")) if dg_type == "windowed" else (0, 0),
                    "radius": get_meta("diameter_deg", 180) / 2,
                }
            elif stim_name == "locally_sparse_noise":
                stim_table = pd.DataFrame(data=data, columns=["start", "end", "frame"])
                stim_table["frame"] = stim_table["frame"].astype(int)

                stim_meta = {
                    "duration": get_meta("duration_sec"),
                    "padding": 0, # time in between successive stimuli
                    "grid_size": 9.3, # get_meta("size_deg") shows up as 9.0 but actually 9.3
                    "num_samples": get_meta("num_samples"),
                }
            elif stim_name == "natural_images" or stim_name == "natural_images_12":
                stim_table = pd.DataFrame(data=data, columns=["start", "end", "image"])
                stim_table["image"] = stim_table["image"].astype(int)
                img_idx_map = get_meta("image_index")
                stim_table["image_index"] = stim_table["image"].apply(lambda image: img_idx_map[image])
                stim_meta = {
                    "duration": get_meta("duration_sec"),
                    "image_description": [s.decode("utf-8") for s in get_meta("image_description")],
                    "image_index": img_idx_map,
                    "n_trials": get_meta("iteration"),
                    "padding": 0, # time in between successive stimuli
                    "num_samples": get_meta("num_samples"),
                }
            elif stim_name == "natural_movie":
                stim_table = pd.DataFrame(data=data, columns=["start", "end", "frame"])
                stim_table["frame"] = stim_table["frame"].astype(int)
                stim_meta = {
                    "num_samples": get_meta("num_samples"),
                }
            elif stim_name == "spontaneous":
                stim_table = pd.DataFrame(data=data, columns=["start", "end", "frame"])
                stim_table.drop("frame", axis=1, inplace=True) # I think frame is useless
                stim_meta = {
                    "duration": get_meta("duration_sec")
                }

        if stim_table is None:
            raise ValueError(f"unable to get stimulus table for {stim_name}")

        # Save in cache
        self._stim_table_cache[stim_name] = (stim_table, stim_meta)
        return stim_table, stim_meta

    def _check_integrity(self, nwb_file):
        error_list = []

        def validate_key(key, loc, desc=None):
            if key not in (nwb_file if loc == "" else nwb_file[loc]):
                if desc is None:
                    error_list.append(f"Cannot find \"{key}\" in \"{loc}\"")
                else:
                    error_list.append(f"Cannot find {desc} (\"{key}\") in \"{loc}\"")
                return False
            return True

        # check vasculature map
        for vm in ['vasmap_mp', 'vasmap_mp_rotated', 'vasmap_wf', 'vasmap_wf_rotated']:
            validate_key(vm, "acquisition/images")

        # check raw time sync data
        for sync in ['digital_2p_vsync',
                    'digital_cam1_exposure',
                    'digital_cam2_exposure',
                    'digital_stim_photodiode',
                    'digital_stim_vsync']:
            validate_key(f"{sync}_rise", "acquisition/timeseries")
            validate_key(f"{sync}_fall", "acquisition/timeseries")

        # check stimulus
        for stim in ['drifting_gratings_full',
                    'drifting_gratings_windowed',
                    'locally_sparse_noise',
                    'natural_images',
                    'natural_images_12',
                    'natural_movie',
                    'spontaneous']:
            validate_key(stim, "stimulus/presentation")

        # check stimulus onset times
        for stim_on_t in ['onsets_drifting_gratings_full',
                        'onsets_drifting_gratings_windowed',
                        'probe_onsets_lsn']:
            validate_key(stim_on_t, "analysis")

        # check eye tracking
        validate_key("eye_tracking_right", "processing")

        # check locomotion
        validate_key("locomotion", "processing")

        # check imaging planes
        rtkeys = [k for k in nwb_file['processing'] if k.startswith('rois_and_traces_plane')]
        # msg += f'\n\tTotal number of imaging plane(s): {len(rtkeys)}.'.expandtabs(4)

        for plane_i in range(len(rtkeys)):
            rtkey = f'rois_and_traces_plane{plane_i}'
            evkey = f'l0_events_plane{plane_i}'

            validate_key(rtkey, "processing")
            validate_key(evkey, "processing")

            seg_path = f'ImageSegmentation/imaging_plane'
            validate_key(seg_path, f"processing/{rtkey}", desc="segmentation results")

            # check traces
            trace_shapes = {}
            for ftkey in ['f_raw',
                        'f_raw_demixed',
                        'f_raw_neuropil',
                        'f_raw_subtracted']:
                if validate_key(f"Fluorescence/{ftkey}", f"processing/{rtkey}", desc="extracted fluorescence traces"):
                    trace_shapes.update({ ftkey: nwb_file[f'processing/{rtkey}/Fluorescence/{ftkey}/data'].shape })

            # check dF/F
            if validate_key("DfOverF/dff_raw", f"processing/{rtkey}", desc="extracted dF/F traces"):
                trace_shapes.update({ "dff_raw": nwb_file[f'processing/{rtkey}/DfOverF/dff_raw/data'].shape })

            # check events
            if validate_key("DfOverF/l0_events", f"processing/{evkey}", desc="extracted events"):
                trace_shapes.update({ 'l0_events': nwb_file[f'processing/{evkey}/DfOverF/l0_events/data'].shape })

            # check projection images
            for ri in ['correlation_projection_denoised',
                    'max_projection_denoised',
                    'max_projection_raw',
                    'mean_projection_denoised',
                    'mean_projection_raw']:

                validate_key(f"reference_images/{ri}", f'processing/{rtkey}/{seg_path}', desc="projection imagings")

            roi_ns = [rk for rk in nwb_file[f'processing/{rtkey}/{seg_path}'].keys() if rk.startswith('roi_') and (not rk.endswith('_list'))]
            roi_ns.sort()
            roi_num = len(roi_ns)

            # msg += f'\n\t\tThis plane has {len(roi_ns)} rois.'.expandtabs(4)

            # print(trace_shapes)

            # check trace shape
            if roi_num == 0: # zero ori
                for t_n, t_shape in trace_shapes.items():
                    if t_shape != (1, 1):
                        error_list.append(f"{t_n} does not have shape (1, 1)")
            else:
                # check roi names
                roi_list = nwb_file[f'processing/{rtkey}/{seg_path}/roi_list'][()]
                roi_list = [r.decode() for r in roi_list]
                if not np.array_equal(np.array(roi_ns), np.array(roi_list)):
                    error_list.append("roi group names and roi_list do not match")

                num_time_sample = []
                for t_n, t_shape in trace_shapes.items():

                    if t_shape[0] != roi_num:
                        error_list.append(f"The first dimension of {t_n} ({t_shape[0]}) does not match the number of rois ({roi_num}).")

                    num_time_sample.append(t_shape[1])

                if len(set(num_time_sample)) != 1:
                    error_list.append(f"Number of timestamps among different traces are not the same ({set(num_time_sample)}).")

            # check prediction shape
            cla_grp_key = f'/analysis/roi_classification_pika/plane{plane_i}'
            
            if validate_key(cla_grp_key, "", desc="pika classification results"):
                cla_grp = nwb_file[cla_grp_key]
                for cla_key in ['pipeline_roi_names',
                                'prediction',
                                'roi_names',
                                'score']:
                    if validate_key(cla_key, cla_grp_key):
                        len_pred = nwb_file[f'{cla_grp_key}/{cla_key}'].shape[0]
                        if len_pred != roi_num:
                            error_list.append(f"The length of prediction results \"{cla_grp_key}/{cla_key}\" ({len_pred}) does not match the number of rois ({roi_num}).")

        if len(error_list) > 0:
            error_str = "\n - ".join(error_list)
            error_str = " - " + error_str
            raise ValueError(f"There are {len(error_list)} errors with the NWB file for session {self.session_id} ({self.nwb_file_path}):\n\n{error_str}")

    def get_mouse_id(self):
        return self.mouse_id
    
    def get_column_id(self):
        return self.column_id

    def get_volume_id(self):
        return self.volume_id

    def get_mouse_column_volume(self):
        return self.mouse_id, self.column_id, self.volume_id

    def get_session_id(self) -> str:
        """Returns the 10-character session ID

        Returns:
            str: 10-character session id 'M<mouse_id>_<column_id><volume_id>' (e.g., 'M409828_13')
        """
        return self.session_id

    def get_lims_session_id(self) -> str:
        """Return the LIMS ophys session id, unique to every nwb file

        Returns:
            str: LIMS ophys session ID for the current session.
        """
        return self.lims_session_id
    
    def get_scope_type(self) -> str:
        """Get the imaging scope used in this imaging session

        Returns:
            str: "2p" for two-photon scope, "3p" for three-photon, error message otherwise.
        """
        return self.scope_type
    
    def get_stim_list(self) -> list:
        """Get the list of stimuli displayed in this session.

        Returns:
            list: List of stimuli names
        """
        return self.stim_list

    def get_vasculature_map(self, type="wf", is_standard=False) -> np.ndarray:
        """_summary_

        Args:
            type (str, optional): "wf" for wild-field, "mp" for multi-photon (2p or 3p). Defaults to "wf".
            is_standard (bool, optional): if False, return the original image; if True, rotate the image to match standard orientation (up: anterior, left: lateral). Defaults to False.

        Returns:
            np.ndarray: 2D array image
        """
        path = f"acquisition/images/vasmap_{type}"

        if is_standard:
            path = f"{path}_rotated"

        with self.open_file() as nwb_file:
            return nwb_file[path][()]

    def get_planes(self) -> list:
        """Get the imaging planes in this session.

        Returns:
            list: Imaging plane IDs in this session
        """
        return self._planes

    def _rois_and_traces(self, plane: int) -> str:
        return f"processing/rois_and_traces_plane{plane}"

    def get_lims_experiment_id(self, plane: int) -> str:
        """Get the LIMS experiment ID for an imaging plane.

        Args:
            plane (int): Plane ID

        Returns:
            str: LIMS experiment ID for this imaging plane
        """
        with self.open_file() as nwb_file:
            return nwb_file[f"{self._rois_and_traces(plane)}/experiment_id"][()].decode()

    def get_plane_depth(self, plane: int) -> int:
        """Get the depth (microns) for an imaging plane.

        Args:
            plane (int): Plane ID

        Returns:
            int: Plane imaging depth (in microns).
        """
        return self._plane_depths[plane]

    def get_plane_projection_images(self, plane: int):
        """Get the projection images for a plane.

        Args:
            plane (str): Plane ID

        Returns:
            proj_raw_mean, proj_raw_max, proj_denoised_mean, proj_denoised_max, proj_denoised_corr; all np.ndarray 2D arrays
        """
        with self.open_file() as nwb_file:
            img_grp = nwb_file[f"{self._rois_and_traces(plane)}/ImageSegmentation/imaging_plane/reference_images"]
            proj_raw_mean = img_grp['mean_projection_raw/data'][()]
            proj_raw_max = img_grp['max_projection_raw/data'][()]
            proj_denoised_mean = img_grp['mean_projection_denoised/data'][()]
            proj_denoised_max = img_grp['max_projection_denoised/data'][()]
            proj_denoised_corr = img_grp['correlation_projection_denoised/data'][()]
        
        return proj_raw_mean, proj_raw_max, proj_denoised_mean, proj_denoised_max, proj_denoised_corr

    def get_rois(self, plane: int) -> list:
        """Get the ROIs in an imaging plane.

        Args:
            plane (int): Plane ID

        Returns:
            list: List of ROIs in the given plane.
        """
        return self._plane_rois[plane]
    
    def get_roi_xy_pixels(self, plane: int, roi: any) -> np.ndarray:
        with self.open_file() as nwb_file:
            plane_grp = nwb_file[f"{self._rois_and_traces(plane)}/ImageSegmentation"]

            pixel_list = []
            if type(roi) in (list, np.array, np.ndarray):
                for r in roi:
                    # rr = str(r)
                    pixels = plane_grp[f"imaging_plane/roi_{r:04}/pix_mask"][()] # pad with leading zeros
                    pixel_list.append(pixels)
                return pixel_list
            else:
                pixels = plane_grp[f"imaging_plane/roi_{roi:04}/pix_mask"][()] # pad with leading zeros
                return pixels


    def get_roi_image_mask(self, plane: int, roi: any) -> np.ndarray:
        """Get the binary ROI image mask of an ROI

        Args:
            plane (int): Plane ID
            roi (int or array of int): ROI ID

        Returns:
            np.ndarray: Binary image mask of the ROI. If given a list of ROIs, the return is the union of all individual masks.
        """
        with self.open_file() as nwb_file:
            plane_grp = nwb_file[f"{self._rois_and_traces(plane)}/ImageSegmentation"]
            width = plane_grp['img_width'][()]
            height = plane_grp['img_height'][()]
            mask = np.zeros((height, width), dtype=bool)
            
            if isinstance(roi, Iterable):
                for r in roi:
                    pixels = plane_grp[f"imaging_plane/roi_{r:04}/pix_mask"][()] # pad with leading zeros
                    mask[pixels[1, :], pixels[0, :]] = True
            else:
                pixels = plane_grp[f"imaging_plane/roi_{roi:04}/pix_mask"][()] # pad with leading zeros
                mask[pixels[1, :], pixels[0, :]] = True


        return mask
    
    def get_pika_roi_ids(self, plane: int) -> list:
        """Get the ROI ids from Pika for a single imaging plane.

        Args:
            plane (int): Plane ID

        Returns:
            list: list of ROI strings, with each entry formatted as "[session_id]_[roi_id]".
            roi_id counting MAY NOT BE CONTINUOUS! Each entry corresponds to an entry from get_rois.
        """
        with self.open_file() as nwb_file:
            roi_grp = nwb_file[f"{self._rois_and_traces(plane)}/ImageSegmentation/imaging_plane"]
            rois = [r for r in roi_grp.keys() if r.startswith("roi_") and not r.endswith("_list")]
            pika_roi_ids = [roi_grp[f"{roi}/roi_description"][()].decode() for roi in rois]
        
        return pika_roi_ids
    
    def get_pika_roi_id(self, plane: int, roi: int) -> str:
        """Get the Pika ROI ID of an ROI.

        Args:
            plane (int): Plane ID
            roi (int): ROI ID

        Returns:
            str: Pika ROI ID, formatted as "[session_id]_[roi_id]".
        """
        with self.open_file() as nwb_file:
            return nwb_file[f"{self._rois_and_traces(plane)}/ImageSegmentation/imaging_plane/roi_{roi:04}/roi_description"][()].decode()
    
    def get_pika_roi_confidence(self, plane: int, roi: any=None):
        """Get the pika classifier confidence score of an ROI. Float between 0 and 1, where 1 indicates full confidence the ROI is a cell soma.

        Args:
            plane (int): Plane ID
            roi (int, list of int, or None): ROI ID(s) or None for all.

        Returns:
            float or list of floats: ROI confidence score, a float between 0 and 1.
        """
        conf = self._plane_pika_roi_conf[plane]
        return conf if roi is None else conf[roi]

    def is_roi_valid(self, plane: int, roi: any=None, conf: float=0.5):
        """Find whether an ROI or list of ROIs is valid (i.e., whether PIKA confidence exceeds a given threshold).

        Args:
            plane (int): Plane ID
            roi (int, list of int, or None): ROI ID(s) or None for all.
            conf (float): Confidence threshold between 0 and 1. Defaults to 0.5.

        Returns:
            bool or np.ndarray: Whether ROI confidence exceeds conf.
        """
        return self.get_pika_roi_confidence(plane, roi) > conf

    def get_traces(self, plane: int, trace_type: str, valid_only: bool=False, reload: bool=False) -> tuple:
        """Get the activity traces in a plane (either all traces or a single trace if an ROI is specified).

        Args:
            plane (int): ID of the imaging plane
            trace_type (str): Type of trace to extract, should be one of "raw", "demixed", "neuropil", "subtracted", "dff", "events"
            valid_only (bool): Mask out rois that are not valid
            reload (bool): Variable to reload data when data is already cached

        Raises:
            LookupError: If trace_type is invalid.

        Returns:
            xr.DataArray of shape (n_rois, n_timestamps). The "roi" coordinate can be used to select ROIs and the "time" coordinate can be used to select time ranges.
        """
        cache_key = (plane, trace_type)

        # Check if cached
        if (cache_key in self._trace_cache) & (~reload):
            return self._trace_cache[cache_key]

        prefix = self._rois_and_traces(plane)

        with self.open_file() as nwb_file:
            if trace_type == "raw":
                trace_grp = nwb_file[f'{prefix}/Fluorescence/f_raw']
            elif trace_type == "demixed":
                trace_grp = nwb_file[f'{prefix}/Fluorescence/f_raw_demixed']
            elif trace_type == 'neuropil':
                trace_grp = nwb_file[f'{prefix}/Fluorescence/f_raw_neuropil']
            elif trace_type == 'subtracted':
                trace_grp = nwb_file[f'{prefix}/Fluorescence/f_raw_subtracted']
            elif trace_type == "dff":
                trace_grp = nwb_file[f'{prefix}/DfOverF/dff_raw']
            elif trace_type == "events":
                # different for events
                trace_grp = nwb_file[f'processing/l0_events_plane{plane}/DfOverF/l0_events']
            elif trace_type == "cascade":
                trace_grp = np.load(f'/home/david.wyrick/projects/V1DD/data/predictions_dff_{self.mouse_id}_{self.column_id}{self.volume_id}_{plane}_all-rois.npz')
            else:
                raise LookupError(f'Do not understand "trace_type", should be one of the following ["raw", "demixed", "neuropil", "sutracted", "dff", "events"]. Got "{trace_type}".')

            trace_data = np.array(trace_grp["data"])
            time = trace_grp["timestamps"][()]

        #Select only valid cells
        rois = np.array(self.get_rois(plane))
        if valid_only:
            mask = self.is_roi_valid(plane)
        else:
            mask = np.ones(trace_data.shape[0],dtype=bool)

        # Save in cache
        traces = xr.DataArray(
            trace_data[mask],
            name=trace_type,
            dims=("roi", "time"),
            coords=dict(roi=rois[mask], time=time)
        )
        self._trace_cache[cache_key] = traces
        
        return traces

    def get_spont_traces(self, plane: int, trace_type: str="events") -> xr.DataArray:
        """Get the spontanenous stimulus response traces for each ROI.

        Args:
            plane (int): Plane ID
            trace_type (str, optional): Type of trace; either "events" or "dff". Defaults to "events".

        Returns:
            xr.DataArray: Array of shape (n_rois, n_spont_frames) containing traces during spontaneous stimulus for each ROI.
        """
        # Pretty sure there is always only one 5min spontaneous stimulus but if this is not true then this needs changing
        table = self.get_stimulus_table("spontaneous")[0]
        start, end = table.at[0, "start"], table.at[0, "end"]

        spont_traces = self.get_traces(plane, trace_type)
        
        return spont_traces.rename(f"spont_{spont_traces.name}") \
            .sel(time=slice(start, end))
    
    def interpolate_all_plane_traces_to_common_time_series(self, interp_method: str="linear", trace_type: str="events",valid_only: bool=True, reload: bool=False) -> xr.DataArray:
        """Get the spontanenous stimulus response traces for each ROI.

        Args:
            common_time_series
            interp_method (str, optional): Type of interpolation method: Defaults to linear
            trace_type (str, optional): Type of trace; either "events" or "dff". Defaults to "events".

        Returns:
            # xr.DataArray: Array of shape (n_rois, n_spont_frames) containing traces during spontaneous stimulus for each ROI.
            list of xr.DataArray
        """
        data_list = []; ts_list = []
        for plane in self.get_planes():
            cache_key = (plane, trace_type)

            # Check if cached
            if (cache_key in self._trace_cache) & (~reload):
                plane_trace = self._trace_cache[cache_key]
            else:
                plane_trace = self.get_traces(plane, trace_type,valid_only=valid_only,reload=True)
            
            data_list.append(plane_trace)
            ts_list.append(plane_trace.coords['time'].values)
        
        #Get "middle" time point
        ts_mean = np.mean(ts_list,axis=0)
        data_list_interp = []
        for ii, plane_trace in enumerate(data_list):
            data_interp = plane_trace.interp(time = ts_mean, method = interp_method, kwargs={'bounds_error':False})
            data_list_interp.append(data_interp)

        # xr.concat(data_list_interp,dim=xr.DataArray(np.arange(1,7),dims='plane'))
        return data_list_interp
            
    def get_running_speed(self):
        """Get the running speed (cm/s) for the current session.

        Returns:
            xr.DataArray containing the running speed, in units of cm/seconds. The "time" coordinate can be used to select time points.
        """
        cache_key = "running_speed" # same across all planes; just so we can cache it

        if cache_key in self._trace_cache:
            return self._trace_cache[cache_key].copy()
        else:
            # Load from NWB file
            with self.open_file() as nwb_file:
                nwb_group = nwb_file["processing/locomotion/Position/distance"]
                timestamps = nwb_group["timestamps"][()] # timestamps, seconds
                total_dist = nwb_group["data"][()] # total distance, cm

                # Compute running speed using discrete derivative
                # The derivative is computed using a central difference approximation
                # The first and last derivatives are computed using a forward and backward difference, respectively
                running_speed = np.empty_like(total_dist)
                running_speed[0] = (total_dist[1] - total_dist[0]) / (timestamps[1] - timestamps[0]) # Forward difference
                running_speed[-1] = (total_dist[-1] - total_dist[-2]) / (timestamps[-1] - timestamps[-2]) # Backward difference
                running_speed[1:-1] = (total_dist[2:] - total_dist[:-2]) / (timestamps[2:] - timestamps[:-2]) # Central difference

            # Save in cache
            run_array = xr.DataArray(running_speed, name="running_speed", coords=dict(time=timestamps))
            self._trace_cache[cache_key] = run_array
            return run_array.copy()
