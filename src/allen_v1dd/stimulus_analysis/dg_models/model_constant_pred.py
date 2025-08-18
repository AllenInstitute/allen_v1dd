import numpy as np
from . import DGModelBase
from .training import DG_STIM_NAMES, get_trial_filter_mask

class DGModelConstantPred(DGModelBase):
    """Simply predicts a mean response, irrespective of stimulus condition"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Since R can't be computed for a constant array (Var(this model predictions) = 0)
        # TODO: How to fix this???
        self.custom_metrics["r"] = lambda y_true, y_pred: np.nan # np.sqrt(skmetrics.r2_score(y_true, y_pred))
    
    def get_trial_label(self, trial_info):
        return 0
    
    def fit(self, X, y):
        self.response_ = np.mean(y)

    def predict(self, X):
        return np.full(len(X), self.response_)
    
    def get_trial_feature_matrix(self, trial_infos, trial_labels):
        # Dummy matrix that is useless
        return np.zeros((len(trial_infos), 1), dtype=int)
    
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
        return self.response_
    
    def set_state(self, state):
        self.response_ = float(state)
    
    def plot_fit(self, axs):
        for ax in axs:
            ax.axhline(y=self.response_, color="black", linestyle="dashed")