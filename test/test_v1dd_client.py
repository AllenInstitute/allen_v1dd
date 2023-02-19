from allen_v1dd.client import OPhysClient

if __name__ == "__main__":
    # client = OPhysClient("/Volumes/programs/mindscope/workgroups/surround/v1dd_in_vivo_new_segmentation/data/")
    client = OPhysClient("/Users/chase/Desktop/test_v1dd_data")

    print(client.get_all_session_ids())

    sess = client.load_ophys_session(mouse=409828, column=1, volume=3)
    
    print("Loaded session", sess.get_session_id())
    print("LIMS session ID", sess.get_lims_session_id())
    print("Imaging planes:")
    for plane in sess.get_planes():
        print(f" - Plane {plane}")
        print(f"    - Imaging depth: {sess.get_plane_depth(plane)} µm")
        print(f"    - Experiment ID: {sess.get_lims_experiment_id(plane)}")
        print(f"    - Total no. ROIs: {len(sess.get_rois(plane))}")
        print(f"    - Valid ROIs: {(sess.is_roi_valid(plane)).sum()}")
    
    
    # client.check_nwb_integrity()