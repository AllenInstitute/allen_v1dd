import numpy as np
import scipy
import pandas as pd


def vonmises_two_peak(x, scale_1, k_1, x0, scale_2, k_2, b):
    # return alpha*scale_1*np.exp(k_1 * np.cos((np.deg2rad(x - x0_1)))) + (1-alpha)*scale_2*np.exp(k_2 * np.cos((np.deg2rad(x - x0_2)))) + b
    return scale_1*np.exp(k_1 * np.cos(np.deg2rad(x - x0))) + scale_2*np.exp(k_2 * np.cos(np.deg2rad(x - x0 - 180))) + b
    # return scale_1*np.exp(k_1 * np.cos((np.deg2rad(2*(x - x0_1)))))

VONMISES_TWO_PEAK_BOUNDS = (
    (0,      0,      0,   0,      0,      0), # lower bounds
    (np.inf, np.inf, 360, np.inf, np.inf, np.inf) # upper bounds
    # s_1     k_1    x0   s_2     k_2      b
)

def vonmises_two_peak_fit(x, y, p0=None, max_fn_calls=(2000, 10000)):
    """Fit params of vonmises_two_peak to best model the (x, y) data.

    Args:
        x (array_like): x values of data points. Must be in degrees!
        y (array_like): y values of data points
        p0 (array_like, optional): Initial point. Defaults to a reasonable starting point.
        max_fn_calls (tuple, optional): Maximum function calls. The next item in the set is used if the previous fails. Defaults to (2000, 20000).

    Returns:
        array_like: Fitted vonmises_two_peak params if the fit was successful, None otherwise.
    """
    if p0 is None:
        p0 = (0.1, 1, 180, 0.01, 1, 0.001) # Seems to be a reasonable starting point
        # p0 = (1e-4, 0.1, 180, 1e-5, 1e-2, 1e-4)
    
    for maxfev in max_fn_calls:
        try:
            params, cov = scipy.optimize.curve_fit(vonmises_two_peak, x, y, maxfev=maxfev, bounds=VONMISES_TWO_PEAK_BOUNDS, p0=p0)
            return params # Return if success
        except RuntimeError as e:
            pass # Absorb errors
    return None # Return None if no fit was successful

def vonmises_two_peak_get_pref_dir(params, return_param_index=False):
    x0 = params[2]
    x1 = (x0 + 180) % 360
    peak_1 = vonmises_two_peak_get_amplitude(x0, params)
    peak_2 = vonmises_two_peak_get_amplitude(x1, params)
    if return_param_index:
        return 0 if peak_1 > peak_2 else 1
    else:
        return x0 if peak_1 > peak_2 else x1

def vonmises_two_peak_get_amplitude(x, params):
    return vonmises_two_peak(x, *params) - params[-1] # f(x) - b

def vonmises_two_peak_get_pref_dir_and_amplitude(params):
    x = vonmises_two_peak_get_pref_dir(params)
    peak = vonmises_two_peak_get_amplitude(x, params)
    return x, peak

def vonmises_two_peak_get_sharpness(params, offset_deg=45):
    x, amp_peak = vonmises_two_peak_get_pref_dir_and_amplitude(params)
    amp_offset = vonmises_two_peak_get_amplitude(x + offset_deg, params)
    return (amp_peak - amp_offset) / (amp_peak + amp_offset)

def vonmises_two_peak_get_peak_k(params):
    i = vonmises_two_peak_get_pref_dir(params, return_param_index=True)
    return params[3 if i == 0 else 5]

def r2_score(y_true, y_pred):
    if type(y_true) is not np.ndarray: y_true = np.array(y_true)
    if type(y_pred) is not np.ndarray: y_pred = np.array(y_pred)
    ss_residuals = float(np.sum(np.square(y_true - y_pred)))
    ss_total = float(np.sum(np.square(y_true - np.mean(y_true))))
    if ss_total == 0:
        return np.nan
    return 1 - (ss_residuals / ss_total)


def get_dg_tuning_params(metrics, roi, dg_type, pref_sf_idx=None):
    if pref_sf_idx is None:
        pref_sf_idx = int(metrics.at[roi, f"{dg_type}_pref_sf_idx"])

    tuning_params = [metrics.at[roi, f"{dg_type}_vonmises_{pref_sf_idx}_param_{i}"] for i in range(6)]
    return tuning_params


def compute_dg_fit_ssi_from_metrics(metrics: pd.DataFrame, metric: str="ssi") -> pd.Series:
    ssi = pd.Series(index=metrics.index, dtype=float)

    for roi in metrics.index:
        try:
            dgw_pref_sf_idx = int(metrics.at[roi, "dgw_pref_sf_idx"])
            dgw_tuning_pref_dir = metrics.at[roi, f"dgw_vonmises_{dgw_pref_sf_idx}_pref_dir"]
            dgw_tuning_peak_amp = metrics.at[roi, f"dgw_vonmises_{dgw_pref_sf_idx}_peak_amp"]
            dgf_tuning_params = get_dg_tuning_params(metrics, roi, "dgf", pref_sf_idx=dgw_pref_sf_idx)
            dgf_tuning_amp_at_dgw_pref_dir = vonmises_two_peak_get_amplitude(dgw_tuning_pref_dir, dgf_tuning_params)
            
            if metric == "ssi":
                ssi[roi] = (dgw_tuning_peak_amp - dgf_tuning_amp_at_dgw_pref_dir) / (dgw_tuning_peak_amp + dgf_tuning_amp_at_dgw_pref_dir)
            elif metric == "si": # 1 - (dgf / max(dgf, dgw))
                ssi[roi] = 1 - (dgf_tuning_amp_at_dgw_pref_dir / max(dgf_tuning_amp_at_dgw_pref_dir, dgw_tuning_peak_amp))
        except:
            ssi[roi] = np.nan

    return ssi

def get_dg_pref_dir(metrics: pd.DataFrame, dg_type: str) -> pd.Series:
    pref_dir = pd.Series(index=metrics.index, dtype=float)

    for roi in metrics.index:
        try:
            dgw_pref_sf_idx = int(metrics.at[roi, "dgw_pref_sf_idx"])
            pref_dir[roi] = metrics.at[roi, f"{dg_type}_vonmises_{dgw_pref_sf_idx}_pref_dir"]
        except:
            pref_dir[roi] = np.nan

    return pref_dir


def get_dg_r2_score(metrics: pd.DataFrame, dg_type: str) -> pd.Series:
    r2_score = pd.Series(index=metrics.index, dtype=float)

    for roi in metrics.index:
        try:
            dgw_pref_sf_idx = int(metrics.at[roi, "dgw_pref_sf_idx"])
            r2_score[roi] = metrics.at[roi, f"{dg_type}_vonmises_{dgw_pref_sf_idx}_r2_score"]
        except:
            r2_score[roi] = np.nan

    return r2_score
