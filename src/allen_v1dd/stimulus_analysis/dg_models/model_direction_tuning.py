import numpy as np
from sklearn.linear_model import LinearRegression
from . import DGModelBase
from .training import DG_STIM_NAMES, get_trial_filter_mask
from . import plotting

class DGModelDirectionTuning(DGModelBase):
    def __init__(self, combine_dg_sizes=False, **kwargs):
        super().__init__(**kwargs)
        self.reg = LinearRegression(fit_intercept=False)
        self.combine_dg_sizes = combine_dg_sizes
        self.n_dimensions = 12 * (1 if combine_dg_sizes else len(DG_STIM_NAMES))
    
    def get_trial_label(self, trial_info):
        label = trial_info["direction"]
        if not self.combine_dg_sizes:
            label += 12*DG_STIM_NAMES.index(trial_info["stim_name"])
        return label
    
    def fit(self, X, y):
        return self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
    
    def get_trial_feature_matrix(self, trial_infos, trial_labels):
        # One-hot encoding of trial labels
        n_trials = len(trial_infos)
        trial_feature_matrix = np.zeros((n_trials, self.n_dimensions), dtype=int)
        trial_feature_matrix[np.arange(n_trials), trial_labels] = 1
        return trial_feature_matrix
    
    def get_roi_train_data(self, group, trial_feature_matrix, trial_infos, trial_labels, trial_responses, roi, group_state_data):
        pref_sf = {
            dg_stim_name: self._get_group_data(group, group_state_data, dg_stim_name, "pref_cond_index")[roi, 1]
            for dg_stim_name in DG_STIM_NAMES
        }
        only_pref_sf = lambda trial_info: trial_info["spatial_frequency"] == pref_sf[trial_info["stim_name"]]
        trial_mask = get_trial_filter_mask(trial_infos, filter=only_pref_sf)
        X = trial_feature_matrix[trial_mask]
        y = trial_responses[trial_mask]
        labels = trial_labels[trial_mask]
        return X, y, labels
    
    def get_state(self):
        return str(list(self.reg.coef_))
    
    def set_state(self, state):
        self.reg.coef_ = state

    def plot_fit(self, axs):
        for i, ax in enumerate(axs):
            tc = self.reg.coef_[i*12:(i+1)*12]
            # np.take with mode=wrap will make 0:13 return items {0, ..., 11, 12, 0}
            ax.plot(plotting.DG_DIRECTIONS, np.take(tc, indices=range(len(plotting.DG_DIRECTIONS)), mode="wrap"), color="black", marker=".")