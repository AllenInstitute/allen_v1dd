import os
from os import path
from datetime import datetime
from traceback import print_exc

import numpy as np
from tqdm import tqdm
import h5py

from allen_v1dd.client import OPhysClient, OPhysSession
from allen_v1dd.stimulus_analysis import *
from allen_v1dd.stimulus_analysis.running_correlation import save_roi_running_correlations
from allen_v1dd.parallel_process import ParallelProcess
from allen_v1dd.duplicate_rois import get_duplicate_roi_pairs_in_session, get_unique_duplicate_rois

def get_h5_group(file, group_path):
    curr_group = file
    for name in group_path:
        if name in curr_group:
            curr_group = curr_group[name]
        else:
            curr_group = curr_group.create_group(name)

    return curr_group



class RunStimulusAnalysis(ParallelProcess):
    def __init__(self, ophys_client, session_ids, stim_analysis_classes, additional_plane_group_tasks, save_dir, task_params):
        super().__init__(save_dir=save_dir)

        self.ophys_client = ophys_client
        self.session_ids = session_ids
        self.stim_analysis_classes = stim_analysis_classes
        self.additional_plane_group_tasks = additional_plane_group_tasks
        self.parent_file_path = path.join(self.get_save_dir(), "stimulus_analyses.h5")
        self.task_params = task_params

    @property
    def test_mode(self):
        return self.task_params is not None and self.task_params.get("test", False)
    
    @property
    def should_debug(self):
        return self.task_params is not None and self.task_params.get("debug", False)


    def start(self):
        """
        Starts the stimulus analysis parallel loading task.

        For every session and plane combination, a task is created to run all stimulus analyses on this session/plane.
        """
        args = []

        for session_id in self.session_ids:
            output_file = path.join(self.get_save_dir(), f"temp_{session_id}.h5")

            # The args here must match the args in the job method
            args.append((self.ophys_client, session_id, self.stim_analysis_classes, self.additional_plane_group_tasks, output_file))

        print(f"There are {len(args)} total planes to process.")
        job_results = self.run(args, parallel=True) # list of (file, group) tuples (return values of job)
        
        # Once jobs are done, merge the temporary files into a single file
        # print(f"Finished jobs. Merging {len(job_results)} files...")

        # with h5py.File(self.parent_file_path, "w") as parent_file:
        #     for temp_file_path, session_group_path in job_results:
        #         dest_group = get_h5_group(parent_file, session_group_path[:-1]) # since the src is copied into the dest group

        #         with h5py.File(temp_file_path, "r") as temp_file:
        #             # Copy the group to the parent file
        #             src_group = get_h5_group(temp_file, session_group_path)
        #             parent_file.copy(source=src_group, dest=dest_group)

        #         # Delete the temp file
        #         if not self.test_mode:
        #             os.remove(temp_file_path)
    
        print(f"Done! Parent file: {self.parent_file_path}")


    def job(self, client: OPhysClient, session_id: str, stim_analysis_classes: list, additional_plane_group_tasks: list, output_file: str):
        """Pre-loads (in parallel) all stimulus analysis information for a given OPhysSession.

        Args:
            client (OPhysClient): OPhys client
            session (str): Session ID for which data should be loaded
            stim_analysis_classes (list): List of StimulusAnalysis subclasses on which to run analyses, formatted as (class, kwargs)
            additional_plane_group_tasks (list): List of methods to run after computing stimulus analyses. Each method has a single argument for the plane h5 group.
            output_file (str): Temporarily file path where outputs are saved

        Returns:
            tuple:
                - temp_output_file (str): Temporary output file path
                - session_group_path (list[str]): Path to session group in the h5 file
        """
        def debug(msg, force=False):
            if force or self.should_debug: print(f"[{session_id}] {msg}")

        debug("Starting job")
        session = client.load_ophys_session(session_id)
        session_group_path = session_id.split("_")

        # Load planes
        planes_to_load = session.get_planes()
        test_max_planes = self.task_params.get("test_max_planes", -1)
        if self.test_mode and test_max_planes > 0:
            planes_to_load = planes_to_load[-test_max_planes:]

        # Load duplicate ROIs
        should_check_dups = len(planes_to_load) > 1
        is_ignored_duplicate = set() # (plane, roi)
        if should_check_dups:
            debug("Loading duplicate ROIs")
            duplicate_roi_pairs = get_duplicate_roi_pairs_in_session(session)
            duplicate_rois = get_unique_duplicate_rois(duplicate_roi_pairs)
            for dup in duplicate_rois:
                for plane_and_roi in dup["plane_and_roi"]:
                    for i in range(len(plane_and_roi)):
                        if i == dup["best_roi_index"]: continue
                        is_ignored_duplicate.add(plane_and_roi)
        
        # Save to file
        with h5py.File(output_file, "w") as file:
            session_group = get_h5_group(file, session_group_path)

            for plane in planes_to_load:
                plane_group = session_group.create_group(f"Plane_{plane}")

                # Load and save stim analyses
                debug(f"Loading and saving stimulus analyses for plane {plane}")
                for SA, sa_kwargs in stim_analysis_classes:
                    analysis = SA(session, plane, **sa_kwargs)
                    analysis_group = plane_group.create_group(analysis.stim_name)
                    try:
                        analysis.save_to_h5(analysis_group)
                    except:
                        debug(f"Error while saving {analysis.stim_name} analyses in {session_id}, plane {plane}:", force=True)
                        print_exc()
                    del analysis # memory thing; maybe not necessary

                # Save general plane information
                plane_group.attrs["session_id"] = session.session_id
                plane_group.attrs["mouse"] = session.mouse_id
                plane_group.attrs["column"] = session.column_id
                plane_group.attrs["volume"] = session.volume_id
                plane_group.attrs["plane"] = plane
                plane_group.attrs["plane_depth_microns"] = session.get_plane_depth(plane)
                plane_group.attrs["date_created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                pika_threshold = 0.5
                is_roi_valid = session.is_roi_valid(plane, conf=pika_threshold)
                n_rois = len(is_roi_valid)
                plane_group.attrs["n_rois"] = n_rois
                plane_group.attrs["n_rois_valid"] = np.count_nonzero(is_roi_valid)

                # Is ROI valid
                ds = plane_group.create_dataset("is_roi_valid", data=is_roi_valid)
                ds.attrs["dimensions"] = ["roi"]
                ds.attrs["pika_threshold"] = pika_threshold

                # Pika ROI score
                ds = plane_group.create_dataset("pika_roi_score", data=session.get_pika_roi_confidence(plane))
                ds.attrs["dimensions"] = ["roi"]

                # ROI image mask centroids
                roi_centroids = np.full((n_rois, 2), np.nan, dtype=float)
                for roi in range(n_rois):
                    if is_roi_valid[roi]:
                        roi_centroids[roi, :] = np.mean(np.where(session.get_roi_image_mask(plane, roi)), axis=1)
                ds = plane_group.create_dataset("roi_centroids", data=roi_centroids)
                ds.attrs["columns"] = ["centroid_row", "centroid_column"]

                # spont_event_traces = session.get_spont_traces(plane, trace_type="events") # shape (n_rois, n_spont_frames)
                # additional_metrics["spontaneous_event_mean"] = spont_event_traces.mean(axis=1) # shape (n_rois,)
                # additional_metrics["spontaneous_event_std"] = spont_event_traces.std(axis=1) # shape (n_rois,)

                # Add is ignored duplicate to each plane group
                plane_is_ignored_duplicate = [
                    (plane, roi) in is_ignored_duplicate
                    for roi in range(n_rois)
                ]
                ds = plane_group.create_dataset("is_ignored_duplicate", data=plane_is_ignored_duplicate)
                plane_group.attrs["n_rois_valid_ignoring_duplicates"] = np.count_nonzero(np.logical_and(is_roi_valid, np.logical_not(plane_is_ignored_duplicate)))

                # Run additional plane tasks
                for task in additional_plane_group_tasks:
                    try:
                        task(session, plane, plane_group)
                    except:
                        debug(f"Error while running additional task {task}", force=True)
                        print_exc()

            # Duplicate ROI information
            group = session_group.create_group("duplicate_rois")
            all_duplicates = []
            if should_check_dups:
                for dup in duplicate_rois:
                    plane_and_roi = dup["plane_and_roi"]
                    best_idx = dup["best_roi_index"]
                    all_duplicates.append(str([plane_and_roi[best_idx]] + plane_and_roi[:best_idx] + plane_and_roi[best_idx+1:]))
            ds = group.create_dataset("all_duplicates", data=all_duplicates)
            ds.attrs["notes"] = "Row format: [(best_plane, best_roi), (plane2, roi2), ...]"

        del session
        debug("Done")

        return output_file, session_group_path

    def output_handler(self, job_result):
        if job_result is None: return
        temp_file_path, session_group_path = job_result

        with h5py.File(self.parent_file_path, "a") as parent_file: # Append to file
            dest_group = get_h5_group(parent_file, session_group_path[:-1]) # since the src is copied into the dest group

            with h5py.File(temp_file_path, "r") as temp_file:
                # Copy the group to the parent file
                src_group = get_h5_group(temp_file, session_group_path)
                parent_file.copy(source=src_group, dest=dest_group)

        # Delete the temp file
        # os.remove(temp_file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="V1DD data directory", type=str)
    parser.add_argument("save_dir", help="Stimulus analysis save directory", type=str, nargs="?")
    parser.add_argument("--test_mode", help="Whether to run in test mode (only run a few sessions)", action="store_true", default=False)
    parser.add_argument("--debug", help="Whether to print debug messages", action="store_true", default=False)
    args = parser.parse_args()

    if args.data_dir == "chase_local":
        base_folder = "/Users/chase/Desktop/test_v1dd_data" # Chase's local data download
        if args.save_dir is None: args.save_dir = "/Users/chase/Desktop/MindScope/allen_v1dd/data_frames/stimulus_analyses"
    elif args.data_dir == "chase_ext":
        base_folder = "/Volumes/ChaseExt/v1dd_ophys_data" # Chase's external hard drive
        if args.save_dir is None: args.save_dir = "/Users/chase/Desktop/MindScope/allen_v1dd/data_frames/stimulus_analyses"
    elif args.data_dir == "isilon":
        base_folder = "/allen/programs/mindscope/workgroups/surround/v1dd_in_vivo_new_segmentation/data"
    else:
        base_folder = args.data_dir

    test_mode = args.test_mode
    debug = args.debug
    task_params = {
        "debug": debug,
        "test": test_mode
    }

    client = OPhysClient(base_folder)

    # Load sessions
    session_ids = client.get_all_session_ids()
    test_max_sessions = task_params.get("test_mode_max_planes", -1)

    if test_mode and test_max_sessions > 0 and len(session_ids) >= test_max_sessions:
        session_ids = session_ids[:test_max_sessions]

    # Test mode: Only M409828 column 1
    # if test_mode: session_ids = [sid for sid in session_ids if sid.startswith("M409828_1")]

    # Test mode: Two sessions from two mice (to test merging)
    if test_mode:
        session_ids = ["M409828_41"]
        task_params["test_max_planes"] = 2

    print(f"Sessions to load ({len(session_ids)}):")
    print(session_ids)

    # List of stimulus analysis classes and their respective kwargs
    stimulus_analysis_classes = [
        (DriftingGratings, dict(dg_type="full", quick_load=test_mode, debug=(debug and test_mode))),
        (DriftingGratings, dict(dg_type="windowed", quick_load=test_mode, debug=(debug and test_mode))),
        (LocallySparseNoise, dict()),
        (NaturalMovie, dict(compute_chisq=False)), # Chi-sq too slow
        (NaturalImages, dict(ns_type="natural_images", compute_chisq=False)),
        (NaturalImages, dict(ns_type="natural_images_12", compute_chisq=False)),
    ]

    # Additional tasks to be run after the stimulus analyses
    # Each task is a method with one argument (the plane h5 group)
    additional_plane_group_tasks = [
        DriftingGratings.compute_ssi_from_h5, # Computes SSI metrics from DGW and DGF analyses
        save_roi_running_correlations,
    ]

    # Process all metrics-loading in parallel
    process = RunStimulusAnalysis(client, session_ids, stimulus_analysis_classes, additional_plane_group_tasks, args.save_dir, task_params)
    process.start()
