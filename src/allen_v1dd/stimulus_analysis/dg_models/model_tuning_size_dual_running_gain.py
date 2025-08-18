import numpy as np

from . import DGModelBase, plotting
from .training import DG_STIM_NAMES, rss_loss, get_trial_filter_mask

class DGModelDirectionTuningSizeDualRunningGain(DGModelBase):
    """
    Models DG responses by fitting a single direction-tuning curve, along with a multiplicative
    gain factor that scales responses to the full-field DG, and TWO multiplicative gain factors
    that scales responses when the mouse is running (one for each DG size).
    """

    def __init__(self, running_threshold=1, **kwargs):
        super().__init__(**kwargs)
        self.weights_ = None # underscore at end to be consistent with sklearn
        self.n_feature_dimensions = 14 # 12 for direction, 1 for size, 1 for locomotion
        self.n_weight_dimensions = 15
        self.n_labels = 48
        self.running_threshold = running_threshold
    
    def _is_running(self, trial_info):
        return abs(trial_info["running_speed"]) >= self.running_threshold

    def get_trial_label(self, trial_info):
        # 0-11 = DGW stationary, 12-23 = DGW running, 24-35 = DGF stationary, 36-47 = DGF running
        return 24*DG_STIM_NAMES.index(trial_info["stim_name"]) + 12*int(self._is_running(trial_info)) + trial_info["direction"]

    def fit(self, X, y):
        def obj_fn(weights):
            y_pred, y_pred_grad = self._predict(X, weights, return_grad=True)
            return rss_loss(y_true=y, y_pred=y_pred, y_pred_grad=y_pred_grad)
            # print(f"Loss: {loss[0]:.4f}")
        
        # Initial guess = mean tuning curve, no size or running modulation
        x0 = np.empty(self.n_weight_dimensions, dtype=float)
        for i in range(12):
            x0[i] = X[X[:, i] == 1].mean()
        x0[12:] = 1 # modulation parameters
        # x0 = np.abs(np.random.randn(self.n_weight_dimensions))
        
        res = self._minimize(obj_fn, x0, method_options=[("CG", dict(maxiter=15000))])
        self.weights_ = res.x # save weights to local model state

    def predict(self, X):
        return self._predict(X, self.weights_) # predict using weights saved in state
    
    def _predict(self, X, weights, return_grad=False):
        if len(X.shape) != 2:
            raise ValueError(f"X must be 2d array, given shape {X.shape}")
        XD, XS, XL = X[:, :12], X[:, -2], X[:, -1] # Xi = [...direction..., size bit, locomotion bit]
        wD, wS, wL0, wL1 = weights[:12], weights[-3], weights[-2], weights[-1]
        dir_pred = XD.dot(wD)
        size_gain = (XS*wS + (1-XS))
        locomotion_gain_if_running = (XS*wL1 + (1-XS)*wL0) # wL0 if windowed, wL1 if full-field
        locomotion_gain = (XL*locomotion_gain_if_running + (1-XL))
        y_pred = dir_pred * size_gain * locomotion_gain
        
        if return_grad:
            d_ypred_wrt_w = np.empty((self.n_weight_dimensions, len(X)))
            d_ypred_wrt_w[:12] = XD.T * size_gain * locomotion_gain # direction
            d_ypred_wrt_w[-3] = dir_pred * locomotion_gain * XS # size
            d_ypred_wrt_w[-2] = dir_pred * size_gain * (XL*(1-XS)) # locomotion 0
            d_ypred_wrt_w[-1] = dir_pred * size_gain * (XL*XS) # locomotion 1
            return y_pred, d_ypred_wrt_w
        else:
            return y_pred
        
    def get_trial_feature_matrix(self, trial_infos, trial_labels):
        # One-hot encoding of trial labels
        n_trials = len(trial_infos)
        trial_feature_matrix = np.zeros(shape=(n_trials, self.n_feature_dimensions), dtype=int)

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

        return X, y, labels
    
    def get_state(self):
        return str(list(self.weights_))
    
    def set_state(self, state):
        self.weights_ = state
    
    def plot_fit(self, axs):
        X = np.zeros((12, self.n_feature_dimensions), dtype=int)
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
        axs[0].text(0.05, 1, f"W Run Gain: {self.weights_[-2]:.2f}", fontsize=12, ha="left", va="top", transform=axs[0].transAxes)
        axs[1].text(0.05, 1, f"Size Gain: {self.weights_[-3]:.2f}", fontsize=12, ha="left", va="top", transform=axs[1].transAxes)
        axs[1].text(1, 1, f"F Run Gain: {self.weights_[-1]:.2f}", fontsize=12, ha="right", va="top", transform=axs[1].transAxes)

        # Make sure text is visible by changing y lim
        ylim = axs[0].get_ylim()
        axs[0].set_ylim(ylim[0], ylim[1]*1.1)