import numpy as np
# from numpy import fft
import scipy
from tqdm.autonotebook import tqdm

from allen_v1dd.client import OPhysSession

def get_duplicate_roi_pairs_in_session(session: OPhysSession, comparison_metric_bounds: dict=dict(centroid_dist_pixels=(0, 30), mask_similarity=(0.4, np.inf), corr_smoothed=(0.45, np.inf)), trace_type: str="dff", gaussian_filter_sigma: float=3) -> list:
    """Get pairs of duplicate ROIs in successive planes in a given ophys session.

    Args:
        session (V1DDOPhysSession): OPhys session
        comparison_metric_bounds (dict, optional): Keys represent metric names, and values are 2-tuples containing lower and upper bounds on the metric value.
                                                   Defaults to mask_similarity ≥ 0.4 and corr_smoothed ≥ 0.45, i.e., dict(mask_similarity=(0.4, np.inf), corr_smoothed=(0.45, np.inf)).
        trace_type (str, optional): Type of traces to compare. Defaults to "dff".
        gaussian_filter_sigma (float, optional): Standard deviation of Gaussian filter applied to traces. Defaults to 3.

    Returns:
        list: List tuples containing information on duplicate ROI pairs. Each tuple is formatted as (plane1, roi1, plane2, roi2, comparison_metrics_dict).
    """
    planes = session.get_planes()

    # Duplicates occur in successive pairs of planes
    if len(planes) == 1:
        return []

    # Used to filter ROIs before comparing traces so not every ROI pair is compared
    # In practice, all ROI pairs that are further than this have no overlap
    trace_stimulus_type = "all" # The full trace from the experiment
    # trace_stimulus_type = "spontaneous" # Only trace during spontaneous stimulus (300 seconds)
    trace_strength_quantile = 0.995
    duplicate_roi_pairs = [] # (plane1, roi1, plane2, roi2, comparison_metrics_dict)

    for plane_i in range(len(planes) - 1):
        plane_1, plane_2 = planes[plane_i], planes[plane_i+1]

        if trace_stimulus_type == "all":
            traces_1 = session.get_traces(plane=plane_1, trace_type=trace_type)
            traces_2 = session.get_traces(plane=plane_2, trace_type=trace_type)
        elif trace_stimulus_type == "spontaneous":
            traces_1 = session.get_spont_traces(plane=plane_1, trace_type=trace_type)
            traces_2 = session.get_spont_traces(plane=plane_2, trace_type=trace_type)
        
        # Ensure traces have same length (dim 1)
        trace_len = min(traces_1.shape[1], traces_2.shape[1])
        traces_1 = traces_1[:, :trace_len]
        traces_2 = traces_2[:, :trace_len]
        traces_1_clean = scipy.ndimage.gaussian_filter1d(traces_1, sigma=gaussian_filter_sigma, axis=1)
        traces_2_clean = scipy.ndimage.gaussian_filter1d(traces_2, sigma=gaussian_filter_sigma, axis=1)
        valid_1 = session.is_roi_valid(plane_1)
        valid_2 = session.is_roi_valid(plane_2)

        roi_valid_matrix = np.outer(valid_1, valid_2)
        n = len(traces_1)
        corr_coefs = np.corrcoef(traces_1, traces_2)[:n, n:]
        corr_coefs_clean = np.corrcoef(traces_1_clean, traces_2_clean)[:n, n:]
        should_compare_matrix = roi_valid_matrix

        if "corr_smoothed" in comparison_metric_bounds:
            lb, ub = comparison_metric_bounds["corr_smoothed"]
            should_compare_matrix = should_compare_matrix & (lb <= corr_coefs_clean) & (corr_coefs_clean <= ub)
        
        if "corr" in comparison_metric_bounds:
            lb, ub = comparison_metric_bounds["corr"]
            should_compare_matrix = should_compare_matrix & (lb <= corr_coefs) & (corr_coefs <= ub)

        for roi_1, roi_2 in zip(*np.where(should_compare_matrix)):
            roi_mask_1 = session.get_roi_image_mask(plane=plane_1, roi=roi_1)
            roi_mask_2 = session.get_roi_image_mask(plane=plane_2, roi=roi_2)
            centroid_1 = np.mean(np.where(roi_mask_1), axis=1)
            centroid_2 = np.mean(np.where(roi_mask_2), axis=1)
            centroid_dist = np.linalg.norm(centroid_1 - centroid_2)
            roi_mask_2 = session.get_roi_image_mask(plane=plane_2, roi=roi_2)
            mask_similarity = np.sum(roi_mask_1 & roi_mask_2) / np.sum(roi_mask_1 | roi_mask_2) # Jaccard similarity = intersection / union

            comparison_metrics = {
                "corr_orig": corr_coefs[roi_1, roi_2],
                "corr_smoothed": corr_coefs_clean[roi_1, roi_2],
                "centroid_dist_pixels": centroid_dist,
                "mask_similarity": mask_similarity,
                "trace_strength_roi_1": np.quantile(traces_1[roi_1], trace_strength_quantile),
                "trace_strength_roi_2": np.quantile(traces_1[roi_2], trace_strength_quantile),
                "mask_size_roi_1": int(np.sum(roi_mask_1)),
                "mask_size_roi_2": int(np.sum(roi_mask_2)),
            }

            # Check if duplicate
            is_duplicate = True
            for metric_key, (lb, ub) in comparison_metric_bounds.items():
                metric_val = comparison_metrics[metric_key]
                if (metric_val < lb) or (metric_val > ub):
                    is_duplicate = False
                    break

            if is_duplicate:
                duplicate_roi_pairs.append((plane_1, roi_1, plane_2, roi_2, comparison_metrics))
    
    return duplicate_roi_pairs


def get_connected_components(graph):
    """Finds connected components in a graph"""
    connected_components = []
    seen = set() # Set of nodes that have been visited
    
    for node in graph:
        if node not in seen:
            # Explore all unseen nodes in graph
            to_explore = set([node])
            connected_component = []

            while len(to_explore) > 0:
                explore_node = to_explore.pop()
                seen.add(explore_node)
                connected_component.append(explore_node)

                # Explore all unseen neighbors of this node
                for neighbor in graph[explore_node]:
                    if neighbor not in seen:
                        to_explore.add(neighbor)
            
            # We have finished exploring a connected component
            connected_components.append(connected_component)

    return connected_components

def get_unique_duplicate_rois(duplicate_roi_pairs: list, best_roi_method: str="trace_strength") -> list:
    """Get a list of all sets of duplicate ROIs from a list of pairs of duplicate ROIs.
    The best ROI is chosen as the one with the largest 2P ROI image mask.

    Args:
        duplicate_roi_pairs (list): Result from get_duplicate_roi_pairs()
        best_roi_method (str): Method to choose the best ROI. One of: "trace_strengtH" (pick one with strongest trace), "mask_size" (largest 2P image mask size)

    Returns:
        list: List of duplicate ROIs. Each entry is a dictionary, with keys:
                plane_and_roi (list): List of (plane, roi) tuple pairs
                best_roi_index (int): Index of the plane_and_roi list that has the "best" ROI (see description above)
    """

    # First build undirected graph of duplicate ROIs (and also keep track of mask size)
    dup_rois_graph = {}
    comparison_values = {}
    comparison_key_1 = f"{best_roi_method}_roi_1"
    comparison_key_2 = f"{best_roi_method}_roi_2"

    for dup in duplicate_roi_pairs:
        # Add bidirectional edge between two nodes
        node1 = dup[0:2] # (plane1, roi1)
        node2 = dup[2:4] # (plane2, roi2)
        if node1 not in dup_rois_graph: dup_rois_graph[node1] = set([])
        if node2 not in dup_rois_graph: dup_rois_graph[node2] = set([])
        dup_rois_graph[node1].add(node2)
        dup_rois_graph[node2].add(node1)
        comparison_values[node1] = dup[4][comparison_key_1]
        comparison_values[node2] = dup[4][comparison_key_2]

    # Find connected components of this graph, indiciating all ROIs that are duplicates
    connected_components = get_connected_components(dup_rois_graph)

    # Merge duplicate ROIs across all planes
    dup_rois_unique = [] # { plane_and_roi: [...], best_index: 0 }

    for plane_and_roi in connected_components:
        plane_and_roi.sort(key=comparison_values.get, reversed=True)
        # roi_mask_sizes = [mask_size[x] for x in plane_and_roi]
        # best_roi_index = int(np.argmax(roi_mask_sizes)) # ROI to keep is the one with the biggest mask

        dup_rois_unique.append(plane_and_roi)

    return dup_rois_unique

def save_duplicates_to_h5(session_group, duplicate_rois, best_roi_method, group_name="duplicate_rois"):
    dup_group = session_group.create_group(group_name)
    all_duplicates = []
    if duplicate_rois is not None:
        for dup_set in duplicate_rois:
            all_duplicates.append(str(dup_set))
    ds = dup_group.create_dataset("all_duplicates", data=all_duplicates)
    ds.attrs["best_roi_method"] = best_roi_method
    ds.attrs["notes"] = "Row format: [(best_plane, best_roi), (plane2, roi2), ...]"

def parse_duplicates_from_h5(session_group):
    return [eval(x.decode()) for x in session_group["duplicate_rois"]["all_duplicates"][()]]