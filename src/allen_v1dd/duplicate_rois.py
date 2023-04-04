import numpy as np
# from numpy import fft
import scipy
# from tqdm.autonotebook import tqdm

from allen_v1dd.client import OPhysSession


def get_duplicate_roi_pairs_in_session(session: OPhysSession, comparison_metric_bounds: dict=dict(mask_similarity=(0.4, None), corr_smoothed=(0.45, None)), trace_type: str="dff", gaussian_filter_sigma: float=3) -> list:
    """Get pairs of duplicate ROIs in successive planes in a given ophys session.

    Args:
        session (V1DDOPhysSession): OPhys session
        comparison_metric_bounds (dict, optional): Keys represent metric names, and values are 2-tuples containing lower and upper bounds on the metric value.
                                                   Defaults to mask_similarity ≥ 0.4 and corr_smoothed ≥ 0.45, i.e., dict(mask_similarity=(0.4, None), corr_smoothed=(0.45, None)).
        trace_type (str, optional): Type of traces to compare. Defaults to "dff".
        gaussian_filter_sigma (float, optional): Standard deviation of Gaussian filter applied to traces. Defaults to 3.

    Returns:
        list: List tuples containing information on duplicate ROI pairs. Each tuple is formatted as (plane1, roi1, plane2, roi2, comparison_metrics_dict).
    """
    planes = session.get_planes()

    # Duplicates occur in successive pairs of planes
    if len(planes) == 1:
        return []

    # Compute 2P ROI centroids for every valid ROI in every plane
    roi_centroids = {
        plane: {
            roi: np.mean(np.where(session.get_roi_image_mask(plane=plane, roi=roi)), axis=1)
            for roi in np.where(session.is_roi_valid(plane=plane))[0]
        }
        for plane in planes
    }

    # Used to filter ROIs before comparing traces so not every ROI pair is compared
    # In practice, all ROI pairs that are further than this have no overlap
    max_pixels_between_centroids = 30
    trace_stimulus_type = "all" # The full trace from the experiment
    # trace_stimulus_type = "spontaneous" # Only trace during spontaneous stimulus (300 seconds)

    duplicate_roi_pairs = [] # (plane1, roi1, plane2, roi2, comparison_metrics_dict)

    for plane_i in range(len(planes) - 1):
        plane_1, plane_2 = planes[plane_i], planes[plane_i+1]

        if trace_stimulus_type == "all":
            all_traces_1 = session.get_traces(plane=plane_1, trace_type=trace_type)
            all_traces_2 = session.get_traces(plane=plane_2, trace_type=trace_type)
        elif trace_stimulus_type == "spontaneous":
            all_traces_1 = session.get_spont_traces(plane=plane_1, trace_type=trace_type)
            all_traces_2 = session.get_spont_traces(plane=plane_2, trace_type=trace_type)
        
        # Ensure traces have same length (dim 1)
        trace_len = min(all_traces_1.shape[1], all_traces_2.shape[1])
        all_traces_1 = all_traces_1[:, :trace_len]
        all_traces_2 = all_traces_2[:, :trace_len]

        # dt = np.mean(np.diff(timestamps_1)) # should be within epsilon across planes
        # fftfreq = fft.rfftfreq(n=trace_len, d=dt)
        # fftfreq_clean_mask = fftfreq < 0.5 # Only low frequencies (tune this parameter)

        for roi_1, centroid_1 in roi_centroids[plane_1].items():
            trace_1 = all_traces_1[roi_1]
            roi_mask_1 = session.get_roi_image_mask(plane=plane_1, roi=roi_1)

            # FFT
            # fft_1 = fft.rfft(trace_1)
            # trace_1_clean = fft.irfft(fft_1 * fftfreq_clean_mask)
            trace_1_clean = scipy.ndimage.gaussian_filter1d(trace_1, sigma=gaussian_filter_sigma)

            for roi_2, centroid_2 in roi_centroids[plane_2].items():
                # If they are far apart, don't bother comparing
                centroid_dist = np.linalg.norm(centroid_1 - centroid_2)
                if centroid_dist > max_pixels_between_centroids:
                    continue
                
                trace_2 = all_traces_2[roi_2]
                roi_mask_2 = session.get_roi_image_mask(plane=plane_2, roi=roi_2)
                mask_similarity = np.sum(roi_mask_1 & roi_mask_2) / np.sum(roi_mask_1 | roi_mask_2) # Jaccard similarity = intersection / union
                corr_coef = np.corrcoef(trace_1, trace_2)[0, 1]

                # fft_2 = fft.rfft(trace_2)
                # trace_2_clean = fft.irfft(fft_2 * fftfreq_clean_mask)
                trace_2_clean = scipy.ndimage.gaussian_filter1d(trace_2, sigma=gaussian_filter_sigma)
                corr_coef_clean = np.corrcoef(trace_1_clean, trace_2_clean)[0, 1]

                comparison_metrics = {
                    "corr_orig": corr_coef,
                    "corr_smoothed": corr_coef_clean,
                    "centroid_dist": centroid_dist,
                    "mask_similarity": mask_similarity,
                    "mask_size_roi_1": int(np.sum(roi_mask_1)),
                    "mask_size_roi_2": int(np.sum(roi_mask_2)),
                }

                # Check if duplicate
                is_duplicate = True
                for metric_key, (lb, ub) in comparison_metric_bounds.items():
                    metric_val = comparison_metrics[metric_key]
                    if (lb is not None and metric_val < lb) or (ub is not None and metric_val > ub):
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

def get_unique_duplicate_rois(duplicate_roi_pairs: list) -> list:
    """Get a list of all sets of duplicate ROIs from a list of pairs of duplicate ROIs.
    The best ROI is chosen as the one with the largest 2P ROI image mask.

    Args:
        duplicate_roi_pairs (list): Result from get_duplicate_roi_pairs()

    Returns:
        list: List of duplicate ROIs. Each entry is a dictionary, with keys:
                plane_and_roi (list): List of (plane, roi) tuple pairs
                best_roi_index (int): Index of the plane_and_roi list that has the "best" ROI (see description above)
    """

    # First build undirected graph of duplicate ROIs (and also keep track of mask size)
    dup_rois_graph = {}
    mask_size = {}
    for dup in duplicate_roi_pairs:
        # Add bidirectional edge between two nodes
        node1 = dup[0:2] # (plane1, roi1)
        node2 = dup[2:4] # (plane2, roi2)
        if node1 not in dup_rois_graph: dup_rois_graph[node1] = set([])
        if node2 not in dup_rois_graph: dup_rois_graph[node2] = set([])
        dup_rois_graph[node1].add(node2)
        dup_rois_graph[node2].add(node1)
        mask_size[node1] = dup[4]["mask_size_roi_1"]
        mask_size[node2] = dup[4]["mask_size_roi_2"]

    # Find connected components of this graph, indiciating all ROIs that are duplicates
    connected_components = get_connected_components(dup_rois_graph)

    # Merge duplicate ROIs across all planes
    dup_rois_unique = [] # { plane_and_roi: [...], best_index: 0 }

    for plane_and_roi in connected_components:
        roi_mask_sizes = [mask_size[x] for x in plane_and_roi]
        best_roi_index = int(np.argmax(roi_mask_sizes)) # ROI to keep is the one with the biggest mask

        dup_rois_unique.append({
            "plane_and_roi": plane_and_roi,
            # "roi_mask_sizes": roi_mask_sizes,
            "best_roi_index": best_roi_index
        })

    return dup_rois_unique