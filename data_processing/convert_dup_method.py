# Associated with Chase's notebook 2023-04-27_2p-duplicates.ipynb
import numpy as np
import h5py
from allen_v1dd.client import OPhysClient
from allen_v1dd.duplicate_rois import parse_duplicates_from_h5, save_duplicates_to_h5, get_ignored_duplicates


def run_isilon():
    client = OPhysClient("isilon")
    filename = "/mnt/drive2/chase/v1dd/v1dd_stim_analyses_2023-04-21_15-57-25/stimulus_analyses.h5"
    new_filename = "/mnt/drive2/chase/v1dd/stimulus_analyses_v2.h5"
    n_total = 0
    n_success = 0
    trace_strength_quantile = 0.995

    with h5py.File(new_filename, "w") as new_file:
        with h5py.File(filename, "r") as old_file:
            for mouse in old_file.keys():
                new_mouse_group = new_file.create_group(mouse)
                for cv in old_file[mouse].keys():
                    old_session_group = old_file[mouse][cv]
                    new_session_group = new_mouse_group.create_group(cv)

                    for plane in old_session_group.keys():
                        if plane == "duplicate_rois":
                            dups = parse_duplicates_from_h5(old_session_group)
                            session_id = f"{mouse}_{cv}"
                            session = client.load_ophys_session(session_id)
                            success = len(dups) == 0 or session is not None
                            best_roi_method = "trace_strength" if success else "mask_size"

                            if session is not None and len(dups) > 0:
                                roi_trace_strengths = {
                                    plane: session.get_traces(plane=plane, trace_type="dff").quantile(trace_strength_quantile, dim="time").values
                                    for plane in session.get_planes()
                                }

                                for dup_set in dups:
                                    dup_set.sort(key=lambda plane_roi: roi_trace_strengths[plane_roi[0]][plane_roi[1]], reverse=True)

                            print(f"Processed {len(dups)} duplicates in {mouse}_{cv} (success = {success})")

                            save_duplicates_to_h5(new_session_group, dups, best_roi_method=best_roi_method)
                            
                            if success: n_success += 1
                            n_total += 1
                        else:
                            new_file.copy(source=old_session_group[plane], dest=new_session_group)

    print(f"{n_success}/{n_total} succesful session copies")


def run_local_fix_is_ignored_dup():
    filename = "/Users/chase/Desktop/stimulus_analyses_v2_BAD_DUP_COUNTS.h5"
    new_filename = "/Users/chase/Desktop/stimulus_analyses_v2.h5"
    n_total = 0
    n_success = 0
    
    with h5py.File(new_filename, "w") as new_file:
        with h5py.File(filename, "r") as old_file:
            for mouse in old_file.keys():
                new_mouse_group = new_file.create_group(mouse)
                for cv in old_file[mouse].keys():
                    old_session_group = old_file[mouse][cv]
                    new_session_group = new_mouse_group.create_group(cv)

                    n_total += 1

                    n_dup_old = 0
                    n_dup_new = 0

                    # Load duplicates from old session group
                    duplicates = parse_duplicates_from_h5(old_session_group)
                    is_ignored_duplicate = get_ignored_duplicates(duplicates)

                    # Update the plane is ignored duplicates
                    for plane_key in old_session_group.keys():
                        old_plane_group = old_session_group[plane_key]

                        # Copy group to new file
                        new_file.copy(source=old_plane_group, dest=new_session_group)
                        new_plane_group = new_session_group[plane_key]
                        
                        # If it is a plane group, update ignored duplicates
                        if "plane" in old_plane_group.attrs:
                            plane = old_plane_group.attrs["plane"]
                            plane_is_ignored_duplicate = np.array([
                                (plane, roi) in is_ignored_duplicate
                                for roi in range(old_plane_group.attrs["n_rois"])
                            ])
                            del new_plane_group["is_ignored_duplicate"]
                            ds = new_plane_group.create_dataset("is_ignored_duplicate", data=plane_is_ignored_duplicate)
                            new_plane_group.attrs["n_rois_valid_ignoring_duplicates"] = np.count_nonzero(np.logical_and(old_plane_group["is_roi_valid"][()], ~plane_is_ignored_duplicate))

                            n_dup_old += old_plane_group.attrs["n_rois_valid_ignoring_duplicates"]
                            n_dup_new += new_plane_group.attrs["n_rois_valid_ignoring_duplicates"]

                    print(f"{mouse}_{cv}", "no change" if n_dup_old == n_dup_new else f"{n_dup_old} --> {n_dup_new}")


                    n_success += 1

    print(f"{n_success}/{n_total} succesful updates")

if __name__ == "__main__":
    # run_isilon()
    run_local_fix_is_ignored_dup()