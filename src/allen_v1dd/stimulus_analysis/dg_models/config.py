import os

import numpy as np
from numpy import nan # so eval() with nan works
import pandas as pd
from scipy import stats as spstats
from sklearn import metrics as skmetrics

from . import training

N_KFOLD_SPLITS = 5
RANDOM_STATE = 42
PLANE_GROUP_FILTER = dict(mouse=409828, column=1)

RUNNING_THRESHOLD = 1

def get_model_config():
    from . import DGModelDirectionTuning, DGModelDirectionTuningSizeGain, DGModelDirectionTuningRunningGain, DGModelDirectionTuningSizeRunningGain, DGModelDirectionTuningSizeDualRunningGain
    return {
        1: {
            "model_class": DGModelDirectionTuning,
            "kwargs": dict(combine_dg_sizes=False),
            "save_file": "model_1.csv",
        },
        2: {
            "model_class": DGModelDirectionTuningSizeGain,
            "kwargs": dict(),
            "save_file": "model_2.csv",
        },
        3: {
            "model_class": DGModelDirectionTuningRunningGain,
            "kwargs": dict(running_threshold=RUNNING_THRESHOLD),
            "save_file": "model_3.csv"
        },
        4: {
            "model_class": DGModelDirectionTuningSizeRunningGain,
            "kwargs": dict(running_threshold=RUNNING_THRESHOLD),
            "save_file": "model_4.csv",
        },
        5: {
            "model_class": DGModelDirectionTuningSizeDualRunningGain,
            "kwargs": dict(running_threshold=RUNNING_THRESHOLD),
            "save_file": "model_5.csv",
        }
    }

MODEL_SAVE_DIRECTORY = "/Users/chase/Desktop/MindScope/allen_v1dd/data_frames/dg_models"
os.makedirs(MODEL_SAVE_DIRECTORY, exist_ok=True)

DG_TYPES = ["windowed", "full"]
DG_STIM_NAMES = [f"drifting_gratings_{dg_type}" for dg_type in DG_TYPES]

MODEL_METRIC_FUNCTIONS = {
    "r": lambda y_true, y_pred: spstats.pearsonr(y_true, y_pred)[0],
    "r2": skmetrics.r2_score,
    "mse": skmetrics.mean_squared_error,
    # "expl_var": skmetrics.explained_variance_score,
}

def instantiate_models(model_config=None):
    if model_config is None: model_config = get_model_config()
    return {
        model_id: config["model_class"](**config.get("kwargs", {}))
        for model_id, config in model_config.items()
    }

def load_evaluation_df(models, plane_group_filter=PLANE_GROUP_FILTER, save_dir=MODEL_SAVE_DIRECTORY, model_config=None):
    if model_config is None: model_config = get_model_config()
    model_df = None

    # Converts the state into a meaningful object (e.g., "[1, 2, 3]" str --> np.array([1, 2, 3]))
    def parse_state(state):
        if pd.isna(state): return state
        state = eval(state)
        if type(state) is list:
            state = np.array(state)
        return state

    for model_id, model in models.items():
        print(f"Loading model {model_id}")
        config = model_config[model_id]
        save_file = os.path.join(save_dir, config["save_file"])
        
        print("  Training if necessary...")
        training.train_and_evaluate(
            model,
            plane_group_filter = plane_group_filter,
            kfold_splitter_fn = training.get_kfold_splitter(),
            suppress_warnings = True,
            save_file = save_file,
            ignore_trained_rois = True,
            build_null_dist = True, null_dist_n_shuffles = 100
        )

        df = pd.read_csv(save_file, index_col="roi_id")
        if "state" in df.columns:
            df["state"] = df["state"].apply(parse_state)
        df.rename(lambda c: f"{c}_model{model_id}", axis="columns", inplace=True) # add identifier to columns

        model_df = df if model_df is None else pd.merge(model_df, df, how="outer", left_index=True, right_index=True)
        print(f"  Trained and evaluated on {len(df)} ROIs")

    return model_df

def load_sig_fit_df(models, model_df, observed_metric="r_test", null_metric="r_valid_null", required_percentile=95):
    # sig_fit_df[roi, model] is true iff sig_fit_observed_metric >= sig_fit_null_metric
    sig_fit_df = pd.DataFrame(index=model_df.index, columns=[f"model{m}" for m in models.keys()], data=np.nan)

    for roi_id in model_df.index:
        for m in models.keys():
            model_key = f"model{m}"
            observed = model_df.at[roi_id, f"{observed_metric}_{model_key}"]
            null_dist = model_df.at[roi_id, f"{null_metric}_{model_key}"]

            if not pd.isna(null_dist):
                null_dist = np.array(eval(null_dist))
                
                # if spstats.percentileofscore(null_dist, observed, nan_policy="omit") >= required_percentile:
                if observed >= np.percentile(null_dist, required_percentile):
                    # The model is responsive; save the observed score
                    sig_fit_df.at[roi_id, model_key] = model_df.at[roi_id, f"r_valid_{model_key}"]

    # Add column for list of responsive models
    def get_model_name(x):
        x = x.values[:len(models)].astype(float)
        models_responsive = np.where(~np.isnan(x))[0] + 1
        return "None" if len(models_responsive) == 0 else ", ".join(map(str, models_responsive))
    sig_fit_df["responsive_models"] = sig_fit_df.apply(get_model_name, axis="columns")
    sig_fit_df.responsive_models.value_counts(normalize=True)

    # Add column for best model
    def get_best_model(x):
        x = x.values[:len(models)].astype(float) # model 1-4 scores
        if np.all(np.isnan(x)):
            return "None"
        else:
            return np.nanargmax(x) + 1
    sig_fit_df["best_model"] = sig_fit_df.apply(get_best_model, axis="columns")

    return sig_fit_df