import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from tqdm import tqdm

from allen_v1dd.client import OPhysClient
from allen_v1dd.stimulus_analysis import DriftingGratings
from allen_v1dd.parallel_process import ParallelProcess
from allen_v1dd.duplicate_rois import get_duplicate_roi_pairs_in_session, get_unique_duplicate_rois


class MetricsLoader(ParallelProcess):
    def __init__(self, sessions, stimulus_analysis_classes):
        super(ParallelProcess, self).__init__()

        self.sessions = sessions
        self.stimulus_analysis_classes = stimulus_analysis_classes

    def get_all_metrics(self):
        # Run the process
        all_metrics = []
        args = []

        for session in sessions:
            for plane in session.get_planes():
                stim_analyses = [SA(session, plane, **kwargs) for SA, kwargs in self.stimulus_analysis_classes]
                args.append((session, plane, stim_analyses, all_metrics))

        print(f"There are {len(args)} total planes to process.")
        return self.run(args, parallel=True)
    
    def job(self, session, plane, stim_analyses, all_metrics):
        is_roi_valid = session.is_roi_valid(plane)
        plane_metrics = []
        dgf = None
        dgw = None

        for stim_analysis in stim_analyses:
            # Save DGW and DGF analyses (used to compute DG metrics)
            if type(stim_analysis) is DriftingGratings:
                if stim_analysis.dg_type == "full":
                    dgf = stim_analysis
                elif stim_analysis.dg_type == "windowed":
                    dgw = stim_analysis

            # Compute metrics
            metrics = stim_analysis.metrics.copy()

            # Drop columns we don't care about since we track them already
            metrics.drop("is_valid", axis=1, inplace=True)

            # Prepend stim label to columns
            # metrics.rename(lambda col: col if col == "is_valid" else f"{stim_analysis.stim_abbrev}_{col}", axis=1, inplace=True)
            metrics.rename(lambda col: f"{stim_analysis.stim_abbrev}_{col}", axis=1, inplace=True)
            plane_metrics.append(metrics)

            # Compute DG surround suppression metrics
            if dgf is not None and dgw is not None:
                ss_metrics = DriftingGratings.compute_ss_metrics_single_plane(dgf, dgw)
                dgf = None
                dgw = None
                # ss_metrics.rename(lambda col: f"dg_{col}", axis=1, inplace=True)
                plane_metrics.append(ss_metrics)

        # Add additional metrics
        additional_metrics = pd.DataFrame()
        spont_event_traces = session.get_spont_traces(plane, trace_type="events") # shape (n_rois, n_spont_frames)
        additional_metrics["spontaneous_event_mean"] = spont_event_traces.mean(axis=1) # shape (n_rois,)
        additional_metrics["spontaneous_event_std"] = spont_event_traces.std(axis=1) # shape (n_rois,)
        plane_metrics.append(additional_metrics)

        # Concatenate all the columns of the DataFrames in plane_metrics
        plane_metrics = pd.concat(plane_metrics, axis=1, join="outer")

        # Add labels to identify plane
        mouse, column, volume = session.get_mouse_column_volume()
        rois = plane_metrics.index # index contains ROI IDs
        plane_metrics.insert(0, "mouse", mouse)
        plane_metrics.insert(1, "column", column)
        plane_metrics.insert(2, "volume", volume)
        plane_metrics.insert(3, "plane", plane)
        plane_metrics.insert(4, "roi", rois)
        plane_metrics.insert(5, "is_valid", is_roi_valid)
        plane_metrics.insert(6, "depth", session.get_plane_depth(plane))

        # Update the index to be unique
        plane_metrics.index = [f"M{mouse}_{column}{volume}_{plane}_{roi}" for roi in rois]

        all_metrics.append(plane_metrics)

        return plane_metrics


if __name__ == "__main__":
    # base_folder = r"\\allen\programs\mindscope\workgroups\surround\v1dd_in_vivo_new_segmentation\data"
    # base_folder = "/Volumes/programs/mindscope/workgroups/surround/v1dd_in_vivo_new_segmentation/data" # if in building, macOS
    # base_folder = "/Volumes/AllenDrive/v1dd/data" # Chase's external hard drive
    base_folder = "/Users/chase/Desktop/test_v1dd_data"
    metrics_save_file = "/Users/chase/Desktop/MindScope/allen_v1dd/data_frames/v1dd_metrics.csv"
    client = OPhysClient(base_folder)

    TEST_MODE = False

    # Load sessions
    mouse = 409828 # selected for EM
    column = 1 # also selected for EM, to avoid duplicate ROIs with columns 2-5
    session_ids = client.get_all_session_ids()
    session_ids = [s for s in session_ids if s.startswith(f"M{mouse}") and s[8] == str(column)]
    print("Sessions to load:")
    print(session_ids)
    sessions = []
    for session_id in tqdm(session_ids):
        session = client.load_ophys_session(session_id=session_id)
        sessions.append(session)
        if TEST_MODE: break
    print(f"Loaded {len(sessions)} ophys sessions")

    # List of stimulus analysis classes and their respecticve kwargs
    stimulus_analysis_classes = [
        (DriftingGratings, dict(dg_type="full")),
        (DriftingGratings, dict(dg_type="windowed")),
    ]
    
    # Process all metrics-loading in parallel
    process = MetricsLoader(sessions, stimulus_analysis_classes)
    all_metrics = process.get_all_metrics()
    all_metrics = pd.concat(all_metrics)

    # Check for duplicate ROIs
    all_metrics.insert(6, "is_ignored_duplicate", False)
    all_metrics.insert(7, "has_duplicate", False)
    all_metrics.insert(8, "n_duplicates", 0)
    all_metrics.insert(9, "duplicate_rois", "")

    for session in tqdm(sessions, desc="Detecting duplicate ROIs..."):
        duplicate_roi_pairs = get_duplicate_roi_pairs_in_session(session)
        duplicate_rois = get_unique_duplicate_rois(duplicate_roi_pairs)
        mouse, column, volume = session.get_mouse_column_volume()

        for dup in duplicate_rois:
            plane_and_roi = dup["plane_and_roi"]
            best_roi_index = dup["best_roi_index"]
            roi_metric_indices = [f"M{mouse}_{column}{volume}_{plane}_{roi}" for plane, roi in plane_and_roi]
            roi_metric_indices.sort()

            for i, roi_metric_index in enumerate(roi_metric_indices):
                # Mark as duplicate
                all_metrics.at[roi_metric_index, "has_duplicate"] = True

                # Ignore if not the best index
                if i != best_roi_index:
                    all_metrics.at[roi_metric_index, "is_ignored_duplicate"] = True

                # Append the other ROIs that are duplicate to this one
                all_metrics.at[roi_metric_index, "n_duplicates"] = len(roi_metric_indices)
                # all_metrics.at[roi_metric_index, "duplicate_rois"] = ", ".join([x for x in roi_metric_indices if x != roi_metric_index])
                all_metrics.at[roi_metric_index, "duplicate_rois"] = ", ".join(roi_metric_indices)

    all_metrics.to_csv(metrics_save_file)
    print(all_metrics)
    