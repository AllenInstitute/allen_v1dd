import numpy as np
from scipy import optimize

from . import DGModelBase, DGModelFailedFitError, plotting
from .training import DG_STIM_NAMES, rss_loss, get_trial_filter_mask

class DGModelDirectionTuningRunningGain(DGModelBase):
    """
    Models DG responses by fitting a single direction-tuning curve, along with a multiplicative
    gain factor that scales responses while the mouse is locomoting.
    """

    def __init__(self, running_threshold=1, **kwargs):
        super().__init__(**kwargs)
        self.weights_ = None # underscore at end to be consistent with sklearn
        self.n_dimensions = 13 # 12 for direction encoding, 1 for size encoding
        self.running_threshold = running_threshold
        
        self._all_dg_X = np.zeros((24, self.n_dimensions), dtype=int)
        i = 0
        for l in range(2):
            for d in range(12):
                self._all_dg_X[i, d] = 1
                self._all_dg_X[i, -1] = l
                i += 1
    
    def _is_running(self, trial_info):
        return abs(trial_info["running_speed"]) >= self.running_threshold

    def get_trial_label(self, trial_info):
        return trial_info["direction"] + 12*int(self._is_running(trial_info))

    def fit(self, X, y):
        # X.shape = (n, d)
        # y.shape = (n,)
        def obj_fn(weights):
            y_pred, y_pred_grad = self._predict(X, weights, return_grad=True)
            return rss_loss(y_true=y, y_pred=y_pred, y_pred_grad=y_pred_grad)
        x0 = np.empty(self.n_dimensions, dtype=float)
        for i in range(12):
            x0[i] = X[X[:, i] == 1].mean()
        x0[-1] = 1
        
        res = self._minimize(obj_fn, x0)
        self.weights_ = res.x # save weights to local model state

    def predict(self, X):
        return self._predict(X, self.weights_) # predict using weights saved in state

    def _predict(self, X, weights, return_grad=False):
        # X.shape = (n, d)
        # weights.shape = (d,)
        if len(X.shape) != 2:
            raise ValueError(f"X must be 2d array, given shape {X.shape}")
        n, d = X.shape
        XD, XL = X[:, :12], X[:, -1]
        wD, wL = weights[:12], weights[-1]
        dir_pred = XD.dot(wD)
        run_gain = (XL*wL + (1-XL))
        y_pred = dir_pred * run_gain
        
        if return_grad:
            d_ypred_wrt_w = np.empty((d, n))
            d_ypred_wrt_w[:12] = XD.T * run_gain
            d_ypred_wrt_w[-1] = dir_pred * XL
            return y_pred, d_ypred_wrt_w
        else:
            return y_pred

    def get_trial_feature_matrix(self, trial_infos, trial_labels):
        # One-hot encoding of trial labels
        n_trials = len(trial_infos)
        trial_feature_matrix = np.zeros(shape=(n_trials, self.n_dimensions), dtype=int)
        trial_feature_matrix[np.arange(n_trials), trial_labels % 12] = 1 # set direction flag
        trial_feature_matrix[np.arange(n_trials), -1] = trial_labels // 12 # set running flag
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
        return str(list(self.weights_))
    
    def set_state(self, state):
        self.weights_ = state
    
    def plot_fit(self, axs):
        X = np.zeros((12, self.n_dimensions), dtype=int)
        np.fill_diagonal(X, 1)
        tc_stat = self.predict(X)
        X[:, -1] = 1 # change from stationary --> running
        tc_run = self.predict(X)

        for i, ax in enumerate(axs):
            ax.plot(plotting.DG_DIRECTIONS, np.take(tc_stat, indices=range(len(plotting.DG_DIRECTIONS)), mode="wrap"), color=plotting.STATIONARY_COLOR, marker=".")
            ax.plot(plotting.DG_DIRECTIONS, np.take(tc_run, indices=range(len(plotting.DG_DIRECTIONS)), mode="wrap"), color=plotting.RUNNING_COLOR, marker=".")
        
        # Plot the gain size
        axs[-1].text(0.05, 1, f"RUN GAIN: {self.weights_[-1]:.2f}", fontsize=12, ha="left", va="top", transform=axs[-1].transAxes)

        # Make sure text is visible by changing y lim
        ylim = axs[0].get_ylim()
        axs[0].set_ylim(ylim[0], ylim[1]*1.1)