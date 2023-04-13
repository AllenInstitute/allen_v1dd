import numpy as np

def get_roi_running_correlations(session, plane, trace_type="dff"):
    """Returns an array of size (n_rois,) of running speed correlations for each ROI.

    Args:
        session (OPhysSession): OPhys session
        plane (int): Plane in session
        trace_type (str, optional): Session trace type. Defaults to "dff".

    Returns:
        np.ndarray: Of length n_rois where each element corresponds to the ith neuron's trace correlation with running speed
    """

    running_speed = session.get_running_speed()
    running_timestamps = running_speed.time.values
    running_speed = running_speed.values

    dff_xr = session.get_traces(plane, trace_type)
    start_time = max(dff_xr.time.values[0], running_timestamps[0])
    end_time = min(dff_xr.time.values[-1], running_timestamps[-1])
    dff_xr = dff_xr.sel(time=slice(start_time, end_time))
    dff = dff_xr.values
    dff_timestamps = dff_xr.time.values

    # Syncing running speed to dF/F timestamps because running speed is sampled at higher res
    run_synced = np.empty(len(dff_timestamps), dtype=float)
    mean_time_window = np.mean(np.diff(dff_timestamps)) / 2

    for i, time in enumerate(dff_timestamps):
        # run_synced[i] = running_speed.sel(time=time, method="nearest").item()
        run_synced[i] = running_speed[(time-mean_time_window <= running_timestamps) & (running_timestamps <= time+mean_time_window)].mean()
    
    corr_mat = np.corrcoef(np.vstack((dff, run_synced))) # corr_mat[i, -1] is the running correlation of ROI i
    roi_run_corr = corr_mat[:-1, -1]
    return roi_run_corr

def save_roi_running_correlations(session, plane, plane_group, trace_type="dff", dataset_name="running_correlations"):
    run_corr = get_roi_running_correlations(session, plane, trace_type=trace_type)
    ds = plane_group.create_dataset(dataset_name, data=run_corr)
    ds.attrs["dimensions"] = ["roi"]
    ds.attrs["trace_type"] = trace_type
    ds.attrs["corr_method"] = "pearson"
