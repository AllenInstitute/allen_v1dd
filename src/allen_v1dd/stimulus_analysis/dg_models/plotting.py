import numpy as np
import matplotlib.pyplot as plt
from allen_v1dd.stimulus_analysis import analysis_tools as at
from allen_v1dd.stimulus_analysis.dg_models import training

DG_DIRECTIONS = np.arange(0, 361, 30)
DG_DIRECTIONS_TICKLABELS = [f"{d}°" for d in DG_DIRECTIONS]
DG_DIRECTIONS_TICKLABELS_SPARSE = [f"{d}°" if di % 3 == 0 else "" for di, d in enumerate(DG_DIRECTIONS)]

RUNNING_COLOR = "b"
RUNNING_COLOR_DARK = "darkblue"
STATIONARY_COLOR = "r"
STATIONARY_COLOR_DARK = "darkred"

def build_figure(nrows=1, figtitle=None, figscale=(6, 4)):
    ncols = len(training.DG_TYPES)
    fig, axs = plt.subplots(figsize=(figscale[0]*ncols, figscale[1]*nrows), ncols=ncols, sharey=True, sharex=True, tight_layout=True)

    for ax, ax_title in zip(axs if nrows == 1 else axs[0], training.DG_TYPES):
        ax.set_title(ax_title)
        ax.set_xticks(ticks=DG_DIRECTIONS, labels=DG_DIRECTIONS_TICKLABELS_SPARSE)

    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=12)

    return fig, axs

def plot_trial_responses(roi_id, axs, color_locomotion=True, scatter_kwargs=dict(facecolor="none", lw=1, s=25)):
    filt, roi = at.plane_group_filter(roi_id)

    for group in at.iter_plane_groups(filter=filt):
        for ax, dg_type in zip(axs, training.DG_TYPES):
            dg = group[f"drifting_gratings_{dg_type}"]
            pref_sf_index = dg["pref_cond_index"][roi, 1]
            trial_responses = dg["trial_responses"][roi, :, pref_sf_index] # shape (n_dir, n_trials)
            trial_running_speeds = np.abs(dg["trial_running_speeds"][:, pref_sf_index]) # same shape ^
            running_trials = trial_running_speeds >= 1
            stationary_trials = trial_running_speeds < 1
            
            for di, d in enumerate(DG_DIRECTIONS):
                di %= 12
                r_stat = trial_responses[di, stationary_trials[di]]
                r_run = trial_responses[di, running_trials[di]]
                ax.scatter([d]*len(r_stat), r_stat, edgecolor=(STATIONARY_COLOR_DARK if color_locomotion else "black"), **scatter_kwargs)
                ax.scatter([d]*len(r_run), r_run, edgecolor=(RUNNING_COLOR_DARK if color_locomotion else "black"), **scatter_kwargs)

    axs[0].set_ylabel("Response (a.u.)", fontsize=12)

def plot_model_fit(models, model_df, model_key, roi_id, axs):
    model = models[model_key]
    state = model_df.at[roi_id, f"state_model{model_key}"]
    model.set_state(state)
    model.plot_fit(axs)