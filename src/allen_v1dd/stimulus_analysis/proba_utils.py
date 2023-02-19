import itertools

import numpy as np
import pandas as pd


def compute_chisq_observed(sweep_responses, sweep_category_matrix):
    """Compute the observed response for each ROI and sweep category.
    Specifically, the observed response for ROI i to category j is defined as
        
        ROI i's mean response to sweeps of category j
         = (ROI i's response to all sweeps)•(binary vector indicating sweeps that are of category j)
              / number of times category j is shown

    We are using matrix multiplications to speed up this computation.    

    Args:
        sweep_responses (np.ndarray): Shape (n_sweeps, n_rois)
        sweep_category_matrix (np.ndarray): Shape (n_sweeps, n_categories)

    Returns:
        np.ndarray: Shape (n_rois, n_categories)
    """

    n_category_trials = sweep_category_matrix.sum(axis=0) # shape (n_categories,)
    mean_category_response = np.dot(sweep_responses.T, sweep_category_matrix) / n_category_trials # shape (n_rois, n_categories)

    # mean_category_response = np.dot(sweep_responses.T, sweep_category_matrix) # shape (n_rois, n_categories)
    # has_category_trial = n_category_trials > 0
    # mean_category_response[:, has_category_trial] /= n_category_trials[has_category_trial]
    # mean_category_response[:, ~has_category_trial] = zero_category_response
    
    return mean_category_response


def compute_chisq_expected(sweep_responses):
    """Compute the expected response for each ROI, defined as
        ROI's mean response to all sweeps

    Args:
        sweep_responses (np.ndarray): Shape (n_sweeps, n_rois)

    Returns:
        np.ndarray: 2d array of shape (n_rois, 1) where each row is a single value indicating the expected ROI response
    """

    # return np.outer(np.mean(sweep_responses, axis=0), n_category_trials)
    return np.expand_dims(np.mean(sweep_responses, axis=0), axis=1)


def compute_chisq_statistic(observed, expected):
    """Compute the chi-square statistic for observed and expected values.

        chi^2 for ROI i = [sum over all categories j] (O_j - E_j) / E_j,
        where O and E denote ROI i's observed and expected responses, respectively, for all categories.

    Args:
        observed (np.ndarray): Shape (n_rois, n_categories)
        expected (np.ndarray): Shape (n_rois, 1)

    Returns:
        np.ndarray: 1d array of shape (n_rois,) containing chi-squared values for each ROI
    """

    chi = np.square(observed - expected) / expected # shape (n_rois, n_categories)
    chi = np.where(expected > 0, chi, 0) # ReLU
    return np.sum(chi, axis=1) # shape (n_rois, )


def get_chisq_sweep_categories(stim_table, stim_table_columns):
    """Return a 

    Args:
        stim_table (pd.DataFrame): DataFrame of length n_sweeps with stimulus information
        stim_table_columns (list): List of columns in stim_table that are used to uniquely define a stimulus category

    Returns:
        tuple:
          - sweep_categories (np.ndarray): 1d integer array of shape (n_sweeps,) indicating the category ID for each sweep
          - sweep_category_matrix (np.ndarray): 2d binary array of shape (n_sweeps, n_categories) where True values in each row indicate when the sweep matches a category
            at the index of the sweep's category.
    """

    stim_table_values = stim_table[stim_table_columns].values
    unique_values = [stim_table[c].dropna().sort_values().unique() for c in stim_table_columns]
    n_sweeps = len(stim_table)
    sweep_categories = np.zeros(n_sweeps, dtype=int)

    # Category for NaN values (if there are any)
    na_mask = stim_table[stim_table_columns].isna().any(axis=1)
    has_na = na_mask.any()
    if has_na:
        sweep_categories[na_mask] = 0 # NaN stim table value corresponds to category 0
    
    # Categories for all other values
    for values in itertools.product(*unique_values): # Get all pairings of stim table values, in sorted order
        mask = np.all(stim_table_values == values, axis=1)
        if np.any(mask): # If there is at least one stimulus presentation of the current pairing
            sweep_categories[mask] = sweep_categories.max() + 1 # Give it a new category (one higher than the previous category)

    if not has_na:
        sweep_categories -= 1 # If no NaN stim table values, then start at zero

    # The resulting n total categories are 0, 1, 2, ..., n-1; each sweep is assigned a category
    n_categories = sweep_categories.max()

    # Compute sweep category matrix
    sweep_category_matrix = np.zeros((n_sweeps, n_categories), dtype=bool)

    for cat in range(n_categories):
        sweep_idx = np.where(sweep_categories == cat)[0]
        sweep_category_matrix[sweep_idx, cat] = True

    return sweep_categories, sweep_category_matrix


def get_chisq_response_proba(stim_table, stim_table_columns, sweep_responses, n_shuffles: int=1000):
    """Returns the chi-squared probability for responsiveness.
    See Dan Millman's paper on contrast enhancement.

    Args:
        stim_table (pd.DataFrame): Stimulus table of length n_sweeps.
        stim_table_columns (list): List of columns used to generate unique stimulus categories.
        sweep_responses (np.ndarray): Matrix of shape (n_sweeps, n_rois).
        n_shuffles (int): Number of times to generate shuffled distribution when computing response probability.

    Returns:
        np.ndarray: 1d array of shape (n_rois,) containing response probabilities for each ROI
    """
    # Compute sweep categories
    sweep_categories, sweep_category_matrix = get_chisq_sweep_categories(stim_table, stim_table_columns)
    categories = np.sort(np.unique(sweep_categories))
    n_categories = len(categories)
    n_sweeps, n_rois = sweep_responses.shape
    
    # Compute observed and expected responses, along with actual chi^2 statistic
    observed = compute_chisq_observed(sweep_responses, sweep_category_matrix)
    expected = compute_chisq_expected(sweep_responses)
    chisq_actual = compute_chisq_statistic(observed, expected)
    
    # Shuffle chi^2
    chisq_shuffle = np.zeros((n_rois, n_shuffles))

    for shuffle_i in range(n_shuffles):
        shuffle_stim_idx = np.random.choice(n_sweeps, size=n_sweeps, replace=True)
        shuffle_sweep_events = sweep_responses[shuffle_stim_idx] # (n_sweeps, n_rois)

        shuffle_observed = compute_chisq_observed(shuffle_sweep_events, sweep_category_matrix)
        shuffle_expected = compute_chisq_expected(shuffle_sweep_events)
        chisq_shuffle[:, shuffle_i] = compute_chisq_statistic(shuffle_observed, shuffle_expected)    
    
    # Compute p-values
    # For ROI i, the p-value is computed as the fraction of times the shuffled chi^2 value is greater than the actual
    p_values = np.mean(chisq_shuffle > np.expand_dims(chisq_actual, axis=1), axis=1)

    return p_values
