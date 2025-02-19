from scipy import optimize

class DGModelFailedFitError(Exception):
    """Error raised when the model failed to fit"""
    def __init__(self, *args, details=None):
        super().__init__(*args)        
        self.details = details

class DGModelBase():
    def __init__(self):
        self.custom_metrics = {}

    def get_roi_inclusion_mask(self, group):
        """Get an ROI inclusion mask that includes ROIs that are valid and non-duplicates.

        Args:
            group (hdf5 file group): Plane group from HDF5 analysis file.

        Returns:
            np.ndarray: Boolean inclusion mask for ROIs in the group
        """
        inclusion = group["is_roi_valid"][()] & ~group["is_ignored_duplicate"][()]
        # dgw = group["drifting_gratings_windowed"]
        # dgf = group["drifting_gratings_full"]
        # inclusion = inclusion & ((dgw["frac_responsive_trials"][()] >= self.frac_resp_trials_thresh) | (dgf["frac_responsive_trials"][()] >= self.frac_resp_trials_thresh))
        return inclusion
    
    def get_trial_label(self, trial_info):
        """Get trial label for a particular trial.
        This method must be implemented when this class is overridden.

        Args:
            trial_info (dict): Info for a trail, e.g., trial_info["direction"], etc.

        Returns:
            int: Trial label for the trial.
        """
        raise NotImplementedError("get_trial_label not implemented")

    def get_trial_feature_matrix(self, trial_infos, trial_labels):
        """Gets a trial feature matrix of shape (n_trials, n_dimensions) for given trials

        Args:
            trial_infos (list): List of trial_info dicts for each trials
            trial_labels (np.ndarray): Labels for each trial

        Raises:
            NotImplemented: _description_
        """
        raise NotImplemented("get_trial_feature_matrix not implemented")

    def get_roi_train_data(self, group, trial_feature_matrix, trial_infos, trial_labels, trial_responses, roi, group_state_data):
        """Train the model on a ROI. If the model is unable to be trained on the ROI,
        this method should return False.

        Args:
            group (hdf5 group): HDF5 file plane group
            trial_feature_matrix: Trial feature matrix, from get_trial_feature_matrix
            trial_infos (list): List of dict trial infos for DG trials
            trial_labels (np.ndarray): 1d array of shape (n_trials,) of trial labels acquired from get_trial_info.
            trial_responses (np.ndarray): 1d array of shape (n_trials,) of trial responses for the given ROI.
            roi (int): ROI number in the plane
            group_state_data (dict): State data for the group that is updated. Used internally for caching HDF5 file data across multiple fits.

        Returns:
            None if the model can't be trained on the ROI, otherwise:
                (X, y, labels) tuple of training data for the ROI.
        """
        raise NotImplementedError("get_roi_train_data not implemented")

    def fit(self, X, y):
        raise NotImplementedError("fit not implemented")
    
    def predict(self, X):
        raise NotImplementedError("predict not implemented")

    def get_state(self):
        return ""
    
    def set_state(self, state):
        pass

    def _get_group_data(self, group, group_state_data, dg_stim_name, key):
        """Helper method to get data from a group, using the given cache.

        Args:
            group (_type_): _description_
            group_state_data (_type_): _description_
            dg_stim_name (_type_): _description_
            key (_type_): _description_

        Returns:
            _type_: _description_
        """
        if dg_stim_name not in group_state_data: group_state_data[dg_stim_name] = {}
        if key not in group_state_data[dg_stim_name]: group_state_data[dg_stim_name][key] = group[dg_stim_name][key][()]
        return group_state_data[dg_stim_name][key]

    def plot_fit(self, axs):
        """Plots the fitted model on a set of axes (one axis for each DG type in training.DG_TYPES).

        Args:
            axs (array like): set of mpl Axis obejcts
        """
        raise NotImplementedError("plot_fit not implemented")
    
    def _minimize(self, fun_with_jac, x0, method_options=[("L-BFGS-B", dict()), ("BFGS", dict(maxiter=10000)), ("CG", dict(maxiter=10000))]):
        """Minimizes an objective function by applying multiple options (stopping when converged).
        Raises DGModelFailedFitError if it cannot converge.

        Args:
            fun_with_jac (callable): Function of a single parameter x that returns a tuple (f(x), f'(x)) where f' is the gradient of f with respect to x.
            x0 (array-like): Initial guess.
            method_options (list, optional): List of (method: str, options: dict) tuples. Defaults to [("L-BFGS-B", dict()), ("BFGS", dict(maxiter=10000)), ("CG", dict(maxiter=10000))].

        Raises:
            DGModelFailedFitError: If convergence isn't met

        Returns:
            Result from scipy.optimize.minimize.
        """
        res = None

        for method, options in method_options:
            # jac=true means function returns (f(x), f'(x))
            res = optimize.minimize(fun=fun_with_jac, x0=x0, jac=True, method=method, options=options)

            # If success, return result; otherwise move on to next options
            if res.success:
                return res
        
        # Failed; raise error
        raise DGModelFailedFitError("Failed to fit", details=res)