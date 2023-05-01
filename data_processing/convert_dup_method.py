# Associated with Chase's notebook 2023-04-27_2p-duplicates.ipynb
import h5py
from allen_v1dd.client import OPhysClient
from allen_v1dd.duplicate_rois import parse_duplicates_from_h5, save_duplicates_to_h5

if __name__ == "__main__":
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