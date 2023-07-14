import numpy as np

from . import DGModelBase, plotting
from .training import DG_STIM_NAMES, rss_loss, get_trial_filter_mask

class DGModelDirectionTuningSizeRunningGain(DGModelBase):
    """
    Models DG responses by fitting a single direction-tuning curve, along with a multiplicative
    gain factor that scales responses to the full-field DG, and a multiplicative gain factor
    that scales responses when the mouse is running.
    """

    def __init__(self, running_threshold=1, **kwargs):
        super().__init__(**kwargs)
        self.weights_ = None # underscore at end to be consistent with sklearn
        self.n_dimensions = 14 # 12 for direction, 1 for size, 1 for locomotion
        self.n_labels = 48
        self.running_threshold = running_threshold
        self._all_dg_X = np.zeros((self.n_labels, self.n_dimensions), dtype=int)
        i = 0
        for s in range(2):
            for l in range(2):
                for d in range(12):
                    self._all_dg_X[i, d] = 1
                    self._all_dg_X[i, -2] = s
                    self._all_dg_X[i, -1] = l
                    i += 1
    
    def _is_running(self, trial_info):
        return abs(trial_info["running_speed"]) >= self.running_threshold

    def get_trial_label(self, trial_info):
        # 0-11 = DGW stationary, 12-23 = DGW running, 24-35 = DGF stationary, 36-47 = DGF running
        return 24*DG_STIM_NAMES.index(trial_info["stim_name"]) + 12*int(self._is_running(trial_info)) + trial_info["direction"]

    def fit(self, X, y):
        def obj_fn(weights):
            y_pred, y_pred_grad = self._predict(X, weights, return_grad=True)
            return rss_loss(y_true=y, y_pred=y_pred, y_pred_grad=y_pred_grad)
        
        # Initial guess = mean tuning curve, no size or running modulation
        x0 = np.empty(self.n_dimensions, dtype=float)
        for i in range(12):
            x0[i] = X[X[:, i] == 1].mean()
        x0[-2] = 1 # size modulation
        x0[-1] = 1 # running modulation
        
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
        XD, XS, XL = X[:, :12], X[:, -2], X[:, -1]
        wD, wS, wL = weights[:12], weights[-2], weights[-1]
        dir_pred = XD.dot(wD)
        size_gain = (XS*wS + (1-XS))
        locomotion_gain = (XL*wL + (1-XL))
        y_pred = dir_pred * size_gain * locomotion_gain
        
        if return_grad:
            d_ypred_wrt_w = np.empty((d, n))
            d_ypred_wrt_w[:12] = XD.T * size_gain * locomotion_gain # direction
            d_ypred_wrt_w[-2] = dir_pred * locomotion_gain * XS # size
            d_ypred_wrt_w[-1] = dir_pred * size_gain * XL # locomotion
            return y_pred, d_ypred_wrt_w
        else:
            return y_pred

    def get_trial_feature_matrix(self, trial_infos, trial_labels):
        # One-hot encoding of trial labels
        n_trials = len(trial_infos)
        trial_feature_matrix = np.zeros(shape=(n_trials, self.n_dimensions), dtype=int)

        for i, trial_info in enumerate(trial_infos):
            trial_feature_matrix[i, trial_info["direction"]] = 1 # set direction flag
            trial_feature_matrix[i, -2] = DG_STIM_NAMES.index(trial_info["stim_name"]) # set size flag
            trial_feature_matrix[i, -1] = int(self._is_running(trial_info)) # set running flag
        
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

        label_counts = np.zeros(self.n_labels, dtype=int)
        for label in labels: label_counts[label] += 1
        # no_samples = label_counts == 0
        # if no_samples[:24].sum() > 0

        # print("LABEL COUNTS", label_counts)

        return X, y, labels
    
    def get_state(self):
        return str(list(self.weights_))
    
    def set_state(self, state):
        self.weights_ = state

    def plot_fit(self, axs):
        X = np.zeros((12, self.n_dimensions), dtype=int)
        np.fill_diagonal(X, 1)

        for i, ax in enumerate(axs):
            X[:, -2] = i # set size flag
            X[:, -1] = 0 # set stationary
            tc_stat = self.predict(X)
            X[:, -1] = 1 # set running
            tc_run = self.predict(X)

            ax.plot(plotting.DG_DIRECTIONS, np.take(tc_stat, indices=range(len(plotting.DG_DIRECTIONS)), mode="wrap"), color=plotting.STATIONARY_COLOR, marker=".")
            ax.plot(plotting.DG_DIRECTIONS, np.take(tc_run, indices=range(len(plotting.DG_DIRECTIONS)), mode="wrap"), color=plotting.RUNNING_COLOR, marker=".")

        # Plot the gain size
        axs[0].text(0.95, 1, f"Run Gain: {self.weights_[-1]:.2f}", fontsize=12, ha="right", va="top", transform=axs[0].transAxes)
        axs[1].text(0.05, 1, f"Size Gain: {self.weights_[-2]:.2f}", fontsize=12, ha="left", va="top", transform=axs[1].transAxes)

        # Make sure text is visible by changing y lim
        ylim = axs[0].get_ylim()
        axs[0].set_ylim(ylim[0], ylim[1]*1.1)