import warnings
import os
from os import path

import numpy as np
from scipy import stats as spstats
from sklearn import metrics as skmetrics
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tqdm.autonotebook import tqdm

from allen_v1dd.stimulus_analysis.analysis_tools import iter_plane_groups, get_roi_id
from . import DGModelBase, DGModelFailedFitError
from .config import *

plane_group_filter = dict(mouse=409828, column=1)

def get_kfold_splitter(plane_group_filter=PLANE_GROUP_FILTER):
    return StratifiedKFold(n_splits=N_KFOLD_SPLITS, shuffle=True, random_state=42).split

def get_trial_infos_and_responses(group):
    # Load trials
    trial_infos = []
    trial_responses = []
    for stim_name in DG_STIM_NAMES:
        for trial_info, responses in iter_trials(group[stim_name], index=True, include_blank=False):
            trial_infos.append(trial_info)
            trial_responses.append(responses)
    trial_responses = np.vstack(trial_responses)
    return trial_infos, trial_responses

TRIAL_RESPONSE_DIMENSIONS = [
    # (dim_name, dim_values)
    ("roi", None),
    ("direction", np.arange(0, 360, 30, dtype=int)),
    ("spatial_frequency", [0.04, 0.08]),
    ("trial", None)
]

def rss_loss(y_true, y_pred, y_pred_grad=None):
    """Compute the residual sum of squares loss, and optionally the loss function gradient.

    Args:
        y_true (np.ndarray): 1d array of (n,) true values
        y_pred (np.ndarray): 1d array of (n,) predicted values
        y_pred_grad (np.ndarray, optional): (d, n) weights where the element at (i, j) represents d(w_i)/d(y_j).
                                            Defaults to None, where no gradient is returned.

    Returns:
        loss if y_pred_grad is None, else (loss, grad) tuple.
    """
    residuals = y_true - y_pred
    loss = np.sum(np.square(residuals))
    if y_pred_grad is None:
        return loss
    else:
        grad = -2 * y_pred_grad.dot(residuals)
        return loss, grad


def iter_trials(dg, index=True, include_blank=False):
    stim_name = dg.attrs["stim_name"]
    trial_responses = dg["trial_responses"][()]
    trial_running_speeds = dg["trial_running_speeds"][()]

    if len(trial_responses) == 0: return
    
    # Identify coordinates where trial is not NaN
    trial_coords = np.vstack(np.where(~np.isnan(trial_responses[0]))).T

    for coords in trial_coords:
        responses = trial_responses[(np.s_[:], *coords)] # (n_rois,); indexing trick akin to [:, coord_0, ..., coord_n]
        index = tuple(coords)
        trial_info = dict(
            stim_name = stim_name,
            index = index,
            running_speed = trial_running_speeds[index]
        )

        for i, coord in enumerate(coords):
            dim_name, dim_values = TRIAL_RESPONSE_DIMENSIONS[1+i]
            trial_info[dim_name] = coord if index or dim_values is None else dim_values[coord]

        yield trial_info, responses
    
    if include_blank:
        blank_responses = dg["blank_responses"][()]

        for trial in range(blank_responses.shape[1]):
            responses = blank_responses[:, trial]
            trial_info = dict(
                stim_name = stim_name,
                blank = True
            )

            for dim_name, dim_values in TRIAL_RESPONSE_DIMENSIONS[1:]:
                trial_info[dim_name] = trial if dim_name == "trial" else -1

            yield trial_info, responses

def get_trial_filter_mask(trial_infos, filter):
    """Get a boolean filter mask where trials are conditionally included based on the given filter.

    Args:
        trial_infos (list): List of trial_infos for each trial
        filter (dict or callable): If dict, then trials are included where trial_info matches filter.
                                   If callable, then trials included where filter(trial_info) is True.

    Returns:
        np.ndarray: Boolean filter mask for trials.
    """
    mask = np.ones(len(trial_infos), dtype=bool)

    for trial, trial_info in enumerate(trial_infos):
        if type(filter) is dict:
            for key, val in filter.items():
                if key not in trial_info: continue
                if type(val) in (list, tuple):
                    if trial_info[key] not in val:
                        mask[trial] = False
                        break
                else:
                    if trial_info[key] != val:
                        mask[trial] = False
                        break
        else:
            mask[trial] = filter(trial_info)
    
    return mask


def train_and_evaluate(model: DGModelBase, plane_group_filter, kfold_splitter_fn, debug=False, suppress_warnings=False, save_file=None, ignore_trained_rois=True, build_null_dist=True, null_dist_n_shuffles=100, random_state=None):
    random = np.random.RandomState(seed=random_state)
    all_df = None

    if ignore_trained_rois and save_file is not None and path.exists(save_file):
        all_df = pd.read_csv(save_file)

    # If ROIs are force set
    new_filt = plane_group_filter.copy() # so we can pop
    if "roi" in plane_group_filter:
        force_rois = new_filt.pop("roi")
        if type(force_rois) is int:
            force_rois = [force_rois]
    else:
        force_rois = None

    for group in iter_plane_groups(filter=new_filt):
        group_identifier = f"{group.attrs['session_id']}_{group.attrs['plane']}"
        trial_infos, trial_responses = get_trial_infos_and_responses(group)
        trial_labels = np.array([model.get_trial_label(trial_info) for trial_info in trial_infos])
        trial_feature_matrix = model.get_trial_feature_matrix(trial_infos, trial_labels)

        # Identify ROIs to include
        rois_to_include = np.where(model.get_roi_inclusion_mask(group))[0] if force_rois is None else force_rois

        # If ignore_trained_rois is True, then remove ROIs that have already been saved
        if ignore_trained_rois and all_df is not None:
            saved_roi_ids = all_df.roi_id.values
            n_rois_before = len(rois_to_include)
            rois_to_include = [
                roi for roi in rois_to_include
                if get_roi_id(group, roi) not in saved_roi_ids
            ]
            n_ignored = n_rois_before - len(rois_to_include)
            if n_rois_before > n_ignored > 0: # if some but not all ROIs were ignored
                print(f"Ignoring {n_ignored}/{n_rois_before} ROIs in {group_identifier}")

        # Stop early if no ROIs in this group
        if len(rois_to_include) == 0:
            continue

        # Prepopulate group state_data; though more may be added to this later depending on model implementation
        group_state_data = {
            dg_stim_name: {
                key: group[dg_stim_name][key][()]
                for key in ("pref_cond_index", "frac_responsive_trials")
            }
            for dg_stim_name in DG_STIM_NAMES
        }

        if debug:
            unq_lbls = np.unique(trial_labels)
            print(len(unq_lbls), "unique labels")

        group_df = []

        for roi in tqdm(rois_to_include, desc=f"{group.attrs['session_id']}_{group.attrs['plane']}", leave=False):
            roi_trial_responses = trial_responses[:, roi]
            _data = model.get_roi_train_data(group, trial_feature_matrix, trial_infos, trial_labels, roi_trial_responses, roi, group_state_data)
            if _data is None: continue # Skip ROI
            X, y, labels = _data
            roi_id = get_roi_id(group, roi)

            if debug:
                print(f"model processing {roi_id}")
                print("labels", pd.value_counts(labels).sort_index())
            
            # Split into K folds
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=UserWarning)
                kfold_splits = list(kfold_splitter_fn(X, labels))

            # Perform cross-validation
            try:
                cv_metrics = cross_validate(model, X, y, kfold_splits, debug=debug)
            except DGModelFailedFitError as e:
                import traceback
                if not suppress_warnings:
                    warnings.warn(f"Failed to fit {get_roi_id(group, roi)}")
                    warnings.warn(str(e.details))
                    traceback.print_exc()
                continue

            # Cross validation succeeded; now compute null metrics
            null_metrics = None
            if build_null_dist:
                try:
                    null_metrics = compute_null_metrics(model, X, y, kfold_splits, n_shuffles=null_dist_n_shuffles, random=random)
                except DGModelFailedFitError as e:
                    import traceback
                    if not suppress_warnings:
                        traceback.print_exc()
                        warnings.warn(f"Failed to compute null metrics for {get_roi_id(group, roi)}: {repr(e)}")
                        warnings.warn(str(e.details))

            if debug:
                print(f"predictions:")
                print(model.predict(X))
                print()

            roi_model_details = dict(
                roi_id = roi_id,
                **cv_metrics,
            )

            if null_metrics is not None:
                for k, v in null_metrics.items():
                    roi_model_details[f"{k}_null"] = v

            group_df.append(roi_model_details)

        # Merge the dataframes and update the file
        group_df = pd.DataFrame(group_df)
        all_df = group_df if all_df is None else all_df.append(group_df)

        # Update save file
        if save_file is not None:
            all_df.to_csv(save_file, index=None)
            print(f"Processed {len(group_df)}/{len(rois_to_include)} ROIs in {group_identifier} ({len(all_df)} total)")
    
    return all_df



def _oops_correct_model_state(model: DGModelBase, plane_group_filter, kfold_splitter_fn, save_file=None):
    all_df = pd.read_csv(save_file, index_col="roi_id")
    saved_rois = all_df.index.values

    for group in iter_plane_groups(filter=plane_group_filter):
        group_identifier = f"{group.attrs['session_id']}_{group.attrs['plane']}"
        trial_infos, trial_responses = get_trial_infos_and_responses(group)
        trial_labels = np.array([model.get_trial_label(trial_info) for trial_info in trial_infos])
        trial_feature_matrix = model.get_trial_feature_matrix(trial_infos, trial_labels)
        rois_to_include = [
            roi for roi in range(group.attrs["n_rois"])
            if get_roi_id(group, roi) in saved_rois
        ]

        # Prepopulate group state_data; though more may be added to this later depending on model implementation
        group_state_data = {
            dg_stim_name: {
                key: group[dg_stim_name][key][()]
                for key in ("pref_cond_index", "frac_responsive_trials")
            }
            for dg_stim_name in DG_STIM_NAMES
        }

        for roi in tqdm(rois_to_include, desc=f"{group.attrs['session_id']}_{group.attrs['plane']}", leave=False):
            roi_trial_responses = trial_responses[:, roi]
            _data = model.get_roi_train_data(group, trial_feature_matrix, trial_infos, trial_labels, roi_trial_responses, roi, group_state_data)
            if _data is None: continue # Skip ROI
            X, y, labels = _data
            roi_id = get_roi_id(group, roi)

            # Perform cross-validation
            try:
                model.fit(X, y)
                all_df.at[roi_id, "state"] = model.get_state()
            except DGModelFailedFitError as e:
                continue

        # Update save file
        if save_file is not None:
            all_df.reset_index().to_csv(save_file, index=None)
    
    return all_df


def _save_fold_metrics(fold_metrics, all_metrics):
    for key, val in fold_metrics.items():
        if key in all_metrics:
            all_metrics[key].append(val)
        else:
            all_metrics[key] = [val]

def _mean_dict_values(d, inplace=False):
    other = d if inplace else {}
    for key in d.keys():
        other[key] = np.mean(d[key], axis=0)
    if not inplace: return other

def cross_validate(model: DGModelBase, X, y, kfold_splits, debug=False):
    """Performs nested cross-validation for model selection and generalizability testing.
    Specifically, data are split into K folds using a given kfold_splitter_fn.
    An outer loop is used to select the a test fold.
    Then an inner loop is used to loop over the remaining K-1 folds to choose a validation fold.
    Finally, the remaining K-2 folds are used as training data.
    The model is fit on the training data and validated on the validation data.
    After the inner loop, the model is trained on the K-1 folds and tested on the test fold.
    All metrics are stored and returned. The mean validation error is used for model selection,
    and the mean test error is used for generalizability error.

    Args:
        model (DGModelBase): Model object with two required methods: fit(X, y), and predict(X)
        X (2d np.ndarray): Training data matrix
        y (1d np.ndarray): Response data for each training trial
        kfold_splits (list): List of k-fold splits

    Returns:
        dict: Dictionary of mean metrics across K folds
    """
    n_splits = len(kfold_splits)
    metrics = {}

    for test_fold, (train_idx, test_idx) in enumerate(kfold_splits):
        valid_true_predict = [] # (y_valid_true, y_valid_predict) for each validation fold

        # Inner cross-validation for model selection
        for valid_fold in range(n_splits):
            if valid_fold == test_fold: continue
            valid_idx = kfold_splits[valid_fold][1] # Validation set (1 fold)

            # Training data is all but test_fold and valid_fold (3 folds)
            inner_train_idx = np.setdiff1d(train_idx, valid_idx)

            if debug: print(f"Starting fold {test_fold}.{valid_fold}")

            # Train model on inner_train_idx
            model.fit(X[inner_train_idx], y[inner_train_idx])
            
            # Test model on valid_idx
            y_valid_true = y[valid_idx]
            y_valid_predict = model.predict(X[valid_idx])

            # Save predictions
            valid_true_predict.append((y_valid_true, y_valid_predict))
        
        # Train model on train_idx
        X_train = X[train_idx]
        y_train_true = y[train_idx]
        model.fit(X_train, y_train_true)
        # y_train_pred = model.predict(X_train)

        # Test model on test_idx
        y_test_true = y[test_idx]
        y_test_pred = model.predict(X[test_idx])

        # Compute fold metrics
        fold_metrics = {}

        for metric_name, metric_fn in MODEL_METRIC_FUNCTIONS.items():
            if metric_name in model.custom_metrics:
                metric_fn = model.custom_metrics[metric_name]
            if metric_fn is None: continue

            fold_metrics[f"{metric_name}_valid"] = np.mean([metric_fn(y_valid_true, y_valid_pred) for y_valid_true, y_valid_pred in valid_true_predict])
            fold_metrics[f"{metric_name}_test"] = metric_fn(y_test_true, y_test_pred)
        
        # Save fold_metrics to metrics
        _save_fold_metrics(fold_metrics, metrics)
    
    # Take mean metrics across folds
    _mean_dict_values(metrics, inplace=True)

    # Train on all data and compute all data metrics
    model.fit(X, y)
    metrics["state"] = model.get_state() # save state after fitting on all training data
    y_pred = model.predict(X)

    for metric_name, metric_fn in MODEL_METRIC_FUNCTIONS.items():
        if metric_name in model.custom_metrics:
            metric_fn = model.custom_metrics[metric_name]
        if metric_fn is None: continue
        metrics[f"{metric_name}_train"] = metric_fn(y, y_pred)
    

    return metrics


def compute_null_metrics(model: DGModelBase, X, y, kfold_splits, n_shuffles=100, random=None):
    if random is None: random = np.random
    null_metrics = {}

    for shuffle in range(n_shuffles):
        # Note we keep the feature matrix (X) and only shuffle the labels (y)
        y_shuffled = np.copy(y)
        random.shuffle(y_shuffled)
        metrics = {}

        for fold, (train_idx, test_idx) in enumerate(kfold_splits):
            X_train, y_train = X[train_idx], y_shuffled[train_idx]
            X_test, y_test = X[test_idx], y_shuffled[test_idx]

            # Fit on data with shuffled responses
            model.fit(X_train, y_train)

            # Evaluate on validation set
            y_test_pred = model.predict(X_test)
            _save_fold_metrics({
                f"{metric_name}_valid": metric_fn(y_test, y_test_pred)
                for metric_name, metric_fn in MODEL_METRIC_FUNCTIONS.items()
            }, metrics)
        
        # Take mean metrics across folds
        _mean_dict_values(metrics, inplace=True)

        _save_fold_metrics(dict(
            # shuffle = shuffle,
            **metrics
        ), null_metrics)
    
    # Take 95 pctile
    for met in list(null_metrics.keys()): # dict can't change size while updating
        null_metrics[f"{met}_95pct"] = np.quantile(null_metrics[met], 0.95)

    return null_metrics