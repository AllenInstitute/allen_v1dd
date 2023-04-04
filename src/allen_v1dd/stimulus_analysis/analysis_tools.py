import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

from . import fit_utils

def set_stylesheet():
    sns.set()
    sns.set_context("paper")
    sns.set_style("ticks")
    mpl.rcParams['axes.linewidth']    = .75
    mpl.rcParams['xtick.major.width'] = .75
    mpl.rcParams['ytick.major.width'] = .75
    mpl.rcParams['xtick.major.size'] = 3
    mpl.rcParams['ytick.major.size'] = 3
    mpl.rcParams['font.size']       = 12
    mpl.rcParams['axes.titlesize']  = 14
    mpl.rcParams['axes.labelsize']  = 14
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False

ANALYSIS_PARAMS = {}

def set_analysis_file(filename):
    ANALYSIS_PARAMS["stim_analysis_filename"] = filename

def set_included_mice(mice_ids=None):
    ANALYSIS_PARAMS["included_mice"] = mice_ids

def set_included_columns(column_ids=None):
    ANALYSIS_PARAMS["included_columns"] = column_ids

def iter_plane_groups(filename: str=None):
    """Iterate all plane groups in an h5 analysis file

    Args:
        filename (str, optional): Filename for h5 analysis file. Defaults to the filename set using set_analysis_file.

    Raises:
        ValueError: If no analysis file is supplied

    Yields:
        h5 group: All plane groups in analysis file
    """
    if filename is None:
        filename = ANALYSIS_PARAMS.get("stim_analysis_filename")
    if filename is None:
        raise ValueError("No stimulus analysis file given. Set one using the set_analysis_file method or the filename argument.")

    mice = ANALYSIS_PARAMS.get("included_mice")
    cols = ANALYSIS_PARAMS.get("included_columns")

    with h5py.File(filename, "r") as file:
        for mouse in file.keys():
            for colvol in file[mouse].keys():
                for plane in file[mouse][colvol]:
                    plane_group = file[mouse][colvol][plane]

                    if "plane" not in plane_group.attrs: continue # Make sure it is actually a plane                    
                    if mice is not None and plane_group.attrs["mouse"] not in mice: continue # Ignore mice
                    if cols is not None and plane_group.attrs["column"] not in cols: continue # # Ignore columns

                    yield plane_group

def load_roi_metrics(metrics_file="../../data_frames/v1dd_metrics.csv", add_columns=True, remove_invalid=True, remove_duplicates=True):
    """Load metrics from the saved CSV file. Default file path is relative so can be accessed from notebooks.

    Args:
        metrics_file (str, optional): Metrics CSV file path. Defaults to "../../dataframes/v1dd_metrics.csv".
        add_columns (bool, optional): Whether to add columns for analyses. Defaults to True.
        remove_invalid (bool, optional): Whether to drop invalid ROIs. Defaults to True.
        remove_duplicates (bool, optional): Whether to drop duplicate ROIs. Defaults to True.
    """

    metrics = pd.read_csv(metrics_file, index_col=0, low_memory=False)

    if remove_duplicates:
        metrics = metrics[metrics.is_valid & ~metrics.is_ignored_duplicate]
    elif remove_invalid:
        metrics = metrics[metrics.is_valid]
    
    if add_columns:
        # Insert new metrics columns for analysis
        metrics["volume"] = metrics["volume"].apply(try_parse_int)
        metrics["depth_chunk"] = metrics["depth"].apply(lambda depth: min(int(np.floor((depth - 50) / 48)), 10)) # 50 is shallowest depth; 48 is size of chunk; deeper chunks go to 10
        metrics["depth_trunc"] = metrics["depth"].apply(lambda depth: int(np.floor(depth / 100) * 100)) # Depth rounded down to 100s
        metrics["vol_plane"] = metrics.apply(lambda row: f"{row.volume}-{row.plane}", axis=1)

        # Compute layer
        # def get_layer_name(depth):
        #     for i, (layer_name, depth_min) in enumerate(zip(V1DDEMClient.LAYER_NAMES, V1DDEMClient.LAYER_BOUNDARIES)):
        #         if depth < depth_min:
        #             # We have found the layer
        #             return layer_name
        #     return "Deep"
        # metrics_all["layer"] = metrics_all.depth.apply(get_layer_name)

        # Drifting gratings
        metrics["dgw_is_responsive"] = metrics.dgw_frac_responsive_trials >= 0.5
        metrics["dgf_is_responsive"] = metrics.dgf_frac_responsive_trials >= 0.5
        metrics["ssi_fit"] = fit_utils.compute_dg_fit_ssi_from_metrics(metrics)
        metrics["dgw_pref_dir_fit"] = fit_utils.get_dg_pref_dir(metrics, "dgw")
        metrics["dgf_pref_dir_fit"] = fit_utils.get_dg_pref_dir(metrics, "dgf")

    return metrics


def plot_dg_tuning_curves(metrics, roi_index, ax=None, figsize=(8, 5), xtick_spacing=30):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(0, 361)
    dgw_pref_sf_idx = int(metrics.at[roi_index, "dgw_pref_sf_idx"])

    for dg_type, color in (("dgw", "red"), ("dgf", "blue")):
        dg_params = fit_utils.get_dg_tuning_params(metrics, roi_index, dg_type)

        if np.any(np.isnan(dg_params)):
            continue

        r2 = metrics.at[roi_index, f"{dg_type}_vonmises_{dgw_pref_sf_idx}_r2_score"]
        ax.plot(x, [fit_utils.vonmises_two_peak(xx, *dg_params) for xx in x], linewidth=2, color=color, label=f"{dg_type.upper()} ($R^2 = {r2:.2f}$)")

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    ax.set_xlim(x.min(), x.max())
    ax.set_xticks(x[::xtick_spacing])
    ax.set_xlabel("Direction (°)")
    ax.set_ylabel("Mean response")
    ssi = metrics.at[roi_index, "ssi"]
    ssi_fit = metrics.at[roi_index, "ssi_fit"]

    ax.set_title(f"{roi_index}\nssi_fit = {ssi_fit:.2f}, ssi = {ssi:.2f}")


def try_parse_int(x):
    try:
        return int(x)
    except:
        return x


def density_scatter(x, y, dtype=float, ax=None, bw_method=None, figsize=(8, 5), cmap="plasma", xlabel=None, ylabel=None, title=None, labelfontsize=14, titlefontsize=16, **kwargs):
    # Based off https://github.com/zhuangjun1981/NeuroAnalysisTools/blob/master/NeuroAnalysisTools/core/PlottingTools.py#L1027

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    
    x = np.array(x).astype(dtype)
    y = np.array(y).astype(dtype)

    # Ensure finite
    include = np.isfinite(x) & np.isfinite(y)
    x, y = x[include], y[include]
    
    # TODO: support for log axes
    xy = np.vstack([x, y])
    kde = st.gaussian_kde(xy, bw_method=bw_method)
    if "s" not in kwargs: kwargs["s"] = 12
    if "alpha" not in kwargs: kwargs["alpha"] = 0.6
    ax.scatter(x, y, c=kde(xy), cmap=cmap, **kwargs)
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=labelfontsize)
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=labelfontsize)
    if title is not None: ax.set_title(title, fontsize=titlefontsize)
    return ax


def heatmap_log_proba_plot(p_matrix, heatmap_labels=None, ticklabels=None, title="Probabilities", titlefontsize=18, xticklabelrotation=90, cbar_label="p", ax=None, ticklabelfontsize=16, figsize=(10, 8), significance_thresh=0.05, log=True, correct_comparisons=True):
    """
    Shorthand to create a probability heatmap matrix.
    heatmap_labels can be used to annotate cells in the matrix.
    tick_labels can be used to set axis tick labels.
    """
    n_comparisons = np.isfinite(p_matrix[0]).sum()
    centervalue = significance_thresh / n_comparisons if correct_comparisons else significance_thresh
    if log:
        centervalue = np.log10(centervalue)
    cbar_ticks = [centervalue-2, centervalue-1, centervalue, centervalue+1, centervalue+2]
    cbar_ticklabels = [f"{x:.0e}" for x in np.power(10, cbar_ticks)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    else:
        fig, ax = ax.get_figure(), ax

    sns.heatmap(
        np.log10(p_matrix), annot=heatmap_labels, fmt="", annot_kws=dict(fontsize=14),
        ax=ax, linewidths=0.5, square=True, cmap="seismic_r",
        center=centervalue, vmin=centervalue-3, vmax=centervalue+3, cbar_kws=dict(ticks=cbar_ticks)
    )

    if ticklabels is not None and p_matrix.shape[1] == len(ticklabels):
        ax.set_xticklabels(ticklabels, fontsize=ticklabelfontsize, rotation=xticklabelrotation)
    # else:
        # ax.set_xticklabels([])

    if ticklabels is not None and p_matrix.shape[0] == len(ticklabels):
        ax.set_yticklabels(ticklabels, fontsize=ticklabelfontsize, rotation=0)
    # else:
    #     ax.set_yticklabels([])

    cbar = ax.collections[0].colorbar
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticklabels)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(cbar_label, fontsize=16) # , rotation=270, va="bottom"
    ax.set_title(title, fontsize=titlefontsize)
    return fig, ax

def dg_response_plot(index, metrics, dg_full, dg_windowed, plot="tuning", **kwargs):
    mouse = metrics.loc[index, "mouse"]
    column = metrics.loc[index, "column"]
    volume = metrics.loc[index, "volume"]
    plane = metrics.loc[index, "plane"]
    roi = metrics.loc[index, "roi"]

    for dgf, dgw in zip(dg_full, dg_windowed):
        if dgf.session.get_mouse_column_volume() == (mouse, column, volume) and dgf.plane == plane:
            if plot == "tuning":
                dgf.plot_roi_tuning_curves(roi, dg_other=dgw, **kwargs)
            elif plot == "condition_response":
                dgw.plot_roi_stim_conditions(roi, **kwargs)
                dgf.plot_roi_stim_conditions(roi, **kwargs)
            elif plot == "trial_response":
                ax1 = dgw.plot_trial_responses(roi, **kwargs)
                ax2 = dgf.plot_trial_responses(roi, **kwargs)
                ylim = (min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1]))
                ax1.set_ylim(ylim)
                ax2.set_ylim(ylim)

            break

