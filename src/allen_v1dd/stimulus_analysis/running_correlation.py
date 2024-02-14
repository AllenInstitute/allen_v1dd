import numpy as np
import xarray as xr

def sync_running_to_response(running_speed_xr, responses_xr, return_xr=True, aggr_method=np.mean):
    # Restrict time window to where both traces have data
    start_time = max(running_speed_xr.time.values[0], responses_xr.time.values[0])
    end_time = min(running_speed_xr.time.values[-1], responses_xr.time.values[-1])
    responses_xr = responses_xr.sel(time=slice(start_time, end_time))

    running_speed = running_speed_xr.values
    running_timestamps = running_speed_xr.time.values
    responses_timestamps = responses_xr.time.values
    mean_time_window = np.mean(np.diff(responses_timestamps)) / 2
    run_synced = np.empty(len(responses_timestamps), dtype=float)

    for i, time in enumerate(responses_timestamps):
        run_synced[i] = aggr_method(running_speed[(time-mean_time_window <= running_timestamps) & (running_timestamps <= time+mean_time_window)])

    if return_xr:
        return xr.DataArray(run_synced, name="synced_running_speed", coords=dict(time=responses_timestamps))
    else:
        return run_synced


def get_roi_running_correlations(session, plane, trace_type="dff"):
    """Returns an array of size (n_rois,) of running speed correlations for each ROI.

    Args:
        session (OPhysSession): OPhys session
        plane (int): Plane in session
        trace_type (str, optional): Session trace type. Defaults to "dff".

    Returns:
        np.ndarray: Of length n_rois where each element corresponds to the ith neuron's trace correlation with running speed
    """

    running_speed_xr = session.get_running_speed()
    running_timestamps = running_speed_xr.time.values
    running_speed = running_speed_xr.values

    dff_xr = session.get_traces(plane, trace_type)
    start_time = max(dff_xr.time.values[0], running_timestamps[0])
    end_time = min(dff_xr.time.values[-1], running_timestamps[-1])
    dff_xr = dff_xr.sel(time=slice(start_time, end_time))
    dff = dff_xr.values
    run_synced = sync_running_to_response(running_speed_xr, dff_xr, return_xr=False)
    corr_mat = np.corrcoef(np.vstack((dff, run_synced))) # corr_mat[i, -1] is the running correlation of ROI i
    roi_run_corr = corr_mat[:-1, -1]
    return roi_run_corr

def save_roi_running_correlations(session, plane, plane_group, trace_type="dff", dataset_name="running_correlations"):
    run_corr = get_roi_running_correlations(session, plane, trace_type=trace_type)
    ds = plane_group.create_dataset(dataset_name, data=run_corr)
    ds.attrs["dimensions"] = ["roi"]
    ds.attrs["trace_type"] = trace_type
    ds.attrs["corr_method"] = "pearson"
