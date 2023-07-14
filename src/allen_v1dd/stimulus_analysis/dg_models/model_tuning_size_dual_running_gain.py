import numpy as np

from . import DGModelDirectionTuningSizeRunningGain, plotting
from .training import rss_loss

class DGModelDirectionTuningSizeDualRunningGain(DGModelDirectionTuningSizeRunningGain):
    """
    Models DG responses by fitting a single direction-tuning curve, along with a multiplicative
    gain factor that scales responses to the full-field DG, and TWO multiplicative gain factors
    that scales responses when the mouse is running (one for each DG size).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weights_ = None # underscore at end to be consistent with sklearn
        self.n_dimensions = 15 # 12 for direction, 1 for size, 2 for locomotion
        self.n_labels = 48
    
    def fit(self, X, y):
        # X.shape = (n, d)
        # y.shape = (n,)
        # print("N UNIQUE ROWS", len(np.unique(X, axis=0)))

        def obj_fn(weights): 
            y_pred, y_pred_grad = self._predict(X, weights, return_grad=True)
            return rss_loss(y_true=y, y_pred=y_pred, y_pred_grad=y_pred_grad)
        
        # Initial guess = mean tuning curve, no size or running modulation
        x0 = np.empty(self.n_dimensions, dtype=float)
        for i in range(12):
            x0[i] = X[X[:, i] == 1].mean()
        x0[12:] = 1 # gain parameters
        
        res = self._minimize(obj_fn, x0=x0)
        self.weights_ = res.x # save weights to local model state

    def _predict(self, X, weights, return_grad=False):
        if len(X.shape) != 2:
            raise ValueError(f"X must be 2d array, given shape {X.shape}")
        XD, XS, XL = X[:, :12], X[:, -2], X[:, -1]
        wD, wS, wL0, wL1 = weights[:12], weights[-3], weights[-2], weights[-1]
        dir_pred = XD.dot(wD)
        size_gain = (XS*wS + (1-XS))
        locomotion_gain_if_running = (XS*wL1 + (1-XS)*wL0) # wL0 if windowed, wL1 if full-field
        locomotion_gain = (XL*locomotion_gain_if_running + (1-XL))
        y_pred = dir_pred * size_gain * locomotion_gain
        
        if return_grad:
            d_ypred_wrt_w = np.empty((self.n_dimensions, len(X)))
            d_ypred_wrt_w[:12] = XD.T * size_gain * locomotion_gain # direction
            d_ypred_wrt_w[-3] = dir_pred * locomotion_gain * XS # size
            d_ypred_wrt_w[-2] = dir_pred * size_gain * (XL*(1-XS)) # locomotion 0
            d_ypred_wrt_w[-1] = dir_pred * size_gain * (XL*XS) # locomotion 1
            return y_pred, d_ypred_wrt_w
        else:
            return y_pred
        
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
        axs[0].text(0.05, 1, f"W Run Gain: {self.weights_[-2]:.2f}", fontsize=12, ha="left", va="top", transform=axs[0].transAxes)
        axs[1].text(0.05, 1, f"Size Gain: {self.weights_[-3]:.2f}", fontsize=12, ha="left", va="top", transform=axs[1].transAxes)
        axs[1].text(1, 1, f"F Run Gain: {self.weights_[-1]:.2f}", fontsize=12, ha="right", va="top", transform=axs[1].transAxes)

        # Make sure text is visible by changing y lim
        ylim = axs[0].get_ylim()
        axs[0].set_ylim(ylim[0], ylim[1]*1.1)