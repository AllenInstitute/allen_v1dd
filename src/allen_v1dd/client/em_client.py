import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from caveclient import CAVEclient
from nglui.statebuilder.helpers import make_neuron_neuroglancer_link, make_synapse_neuroglancer_link
from standard_transform import v1dd_transform_nm
from pcg_skel import coord_space_meshwork
from meshparty.meshwork.meshwork import Meshwork


CAVE_DATASTACK_NAME = "v1dd"
CAVE_SERVER_ADDRESS = "https://globalv1.em.brain.allentech.org"

# LAYER_BOUNDARIES = [200, 375, 500, 680, 850]
# LAYER_NAMES = ["1", "2/3", "4", "5", "6"]


DEFAULT_VOXEL_RESOLUTION = [1, 1, 1]


def cave_client_setup(make_new=False, token=None):
    cave_client = CAVEclient(datastack_name=CAVE_DATASTACK_NAME, server_address=CAVE_SERVER_ADDRESS)

    if token is None:
        if make_new:
            print("NOTE: Make sure you are signed into your @alleninstitute.org email!!!")
        cave_client.auth.setup_token(make_new=make_new)
    else:
        cave_client.auth.save_token(token=token, overwrite=True)
        print("Set auth token:", token)

class EMClient:
    def __init__(self):
        self.cave_client = CAVEclient(datastack_name=CAVE_DATASTACK_NAME, server_address=CAVE_SERVER_ADDRESS)
        self.cave_client_seg_cv = self.cave_client.info.segmentation_cloudvolume(progress=True, parallel=1)
        self._transform_nm_to_microns = v1dd_transform_nm() # Transform from nm to oriented microns
        self._neuron_meshwork_cache = {} # root_id -> meshparty.meshwork.meshwork.Meshwork
        
        self.voxel_resolution_attr = "dataframe_resolution"
        self.nucleus_table = "nucleus_detection_v0"
        self.dendrite_proofreading_table = "ariadne_dendrite_task"
        self.axon_proofreading_table = "ariadne_axon_task"
        self.proofreading_status_none = "not_started"
        self.proofreading_status_complete = ["clean", "complete", "submitted"]
        self.neural_cell_types = ["PYC", "MC", "BC", "BPC", "NGC"]

        self.layer_separations = [-np.inf, 100, 270, 400, 550, 750]
        self.layer_names = ["1", "2/3", "4", "5", "6"]
        self.layer_boundaries = {
            layer: (self.layer_separations[i], self.layer_separations[i+1])
            for i, layer in enumerate(self.layer_names)
        } # {"1": (-np.inf, 100), "2/3": (100, 270), "4": (270, 400), "5": (400, 550), "6": (550, 750)}

        # Shorthands
        self.materialize = self.cave_client.materialize
        self.version = self.cave_client.materialize.version
        self.get_tables = self.cave_client.materialize.get_tables
        self.query_table = self.cave_client.materialize.query_table
        self.get_table_metadata = self.cave_client.materialize.get_table_metadata
        self.synapse_table = self.cave_client.materialize.synapse_table


    def get_voxel_res(self, df: pd.DataFrame):
        if self.voxel_resolution_attr in df.attrs:
            res = df.attrs[self.voxel_resolution_attr]
            if res == "mixed_resolutions":
                return DEFAULT_VOXEL_RESOLUTION
            else:
                return res
        
        return DEFAULT_VOXEL_RESOLUTION


    def transform_position_to_microns(self, position, voxel_res=None, df=None):
        if df is not None:
            voxel_res = self.get_voxel_res(df) # nm / voxel
            
        if voxel_res is not None:
            position = position * voxel_res # convert voxels -> nm
        else:
            # Assume position is already in nm
            pass
        
        transformed_position = self._transform_nm_to_microns.apply(position) # nm -> transformed microns
        transformed_position = transformed_position.reshape(position.shape) # Maintain original shape
        return transformed_position

    
    def df_position_to_microns(self, df, position_column, new_position_column=None):
        df[new_position_column] = df[position_column].apply(lambda pos: self.transform_position_to_microns(pos, df=df))


    def get_single_soma_position(self, pt_root_id, units="voxels", ignore_multi=False, return_voxel_resolution=False):
        soma_query_result = self.query_table(self.nucleus_table, filter_equal_dict=dict(pt_root_id=pt_root_id))
        if len(soma_query_result) > 1 and not ignore_multi:
            print(f"WARN: pt_root_id={pt_root_id} has multiple ({len(soma_query_result)} soma results; choosing first")

        soma_pos_voxels = soma_query_result.pt_position.iloc[0]

        if units == "voxels":
            pos = soma_pos_voxels
        elif units == "microns":
            pos = self.transform_position_to_microns(self, soma_pos_voxels, df=soma_query_result)
        else:
            raise ValueError(f"bad units: {units}")

        if return_voxel_resolution:
            return pos, self.get_voxel_res(soma_query_result)
        else:
            return pos


    def get_soma_positions(self, pt_root_ids):
        if type(pt_root_ids) is int:
            somas = self.query_table(self.nucleus_table, filter_equal_dict=dict(pt_root_id=pt_root_ids))
        else:
            somas = self.query_table(self.nucleus_table, filter_in_dict=dict(pt_root_id=pt_root_ids))
        somas.drop_duplicates("pt_root_id", inplace=True)
        self.df_position_to_microns(somas, "pt_position", "position_microns")
        return somas

    def get_soma_position_microns_map(self, pt_root_ids):
        soma_table = self.get_soma_positions(pt_root_ids)
        soma_map = pd.Series(index=soma_table.pt_root_id.values, data=soma_table.position_microns.values) # pt_root_id -> position_microns
        return soma_map
    
    def get_cells_in_box(self, min_voxels, max_voxels, drop_duplicates=False):
        nuclei_in_box = self.query_table(self.nucleus_table, filter_spatial_dict=dict(pt_position=[min_voxels, max_voxels]))
        
        if drop_duplicates:
            nuclei_in_box.drop_duplicates("pt_root_id", keep=False, inplace=True)
            nuclei_in_box.reset_index(inplace=True)
        
        self.df_position_to_microns(nuclei_in_box, "pt_position", "position_microns")

        return nuclei_in_box

    def get_nearby_cells(self, pt_root_id=None, center_position_voxels=None, radius_microns=100, box_type="sphere", drop_duplicates=False):
        if pt_root_id is not None:
            center_position_voxels = self.get_single_soma_position(pt_root_id, units="voxels")
        elif center_position_voxels is not None:
            center_position_voxels = np.array(center_position_voxels)
            
        if center_position_voxels is None:
            raise ValueError("no center position given")

        nm_per_voxel = np.array([1, 1, 1], dtype=int) # this is the voxel resolution for nucleus detection table
        radius_voxels = nm_per_voxel * radius_microns * 1000
        min_pos = center_position_voxels - radius_voxels
        max_pos = center_position_voxels + radius_voxels
        nuclei_in_box = self.get_cells_in_box(min_voxels=min_pos, max_voxels=max_pos, drop_duplicates=drop_duplicates)
        center_position_microns = self.transform_position_to_microns(center_position_voxels, df=nuclei_in_box) # same table, same voxel resolution

        axes = [0, 2] if box_type == "cylinder" else [0, 1, 2] # ignore y distance for cylinder
        dist_to_center = nuclei_in_box["position_microns"].apply(lambda pos: np.linalg.norm(pos[axes] - center_position_microns[axes]))
        nuclei_in_box["dist_to_center"] = dist_to_center

        if box_type == "sphere" or box_type == "cylinder":
            nuclei_in_box = nuclei_in_box[dist_to_center <= radius_microns]
        elif box_type == "cube" or box_type == "box":
            # The cube was already defined in the original query
            pass
        else:
            raise ValueError(f"bad box_type: {box_type}")

        nuclei_in_box.reset_index(inplace=True)
        return nuclei_in_box


    def _get_synapses(self, pre_ids=None, post_ids=None, microns_position_mappings=None, soma_position_mappings=dict(pre_pt_root_id="pre_soma_position", post_pt_root_id="post_soma_position")):
        synapses = self.materialize.synapse_query(pre_ids=pre_ids, post_ids=post_ids)
        synapses.reset_index(inplace=True) # Make from 0, 1, ..., n-1

        if microns_position_mappings is not None:
            for position_column, new_position_column in microns_position_mappings.items():
                self.df_position_to_microns(synapses, position_column, new_position_column)
        
        if soma_position_mappings is not None:
            # Find unique pt_root_ids that we need for the soma position mapping
            unique_pt_root_ids = set()
            for pt_root_id_col, new_position_column in soma_position_mappings.items():
                unique_pt_root_ids.update(synapses[pt_root_id_col].unique())
            unique_pt_root_ids = list(unique_pt_root_ids)

            # Find the soma positions
            soma_table = self.get_soma_positions(unique_pt_root_ids)
            soma_positions_voxels = pd.Series(index=soma_table.pt_root_id.values, data=soma_table.pt_position.values)
            soma_positions_microns = pd.Series(index=soma_table.pt_root_id.values, data=soma_table.position_microns.values)

            # Add new column(s) mapping the pt_root_id to soma position (in microns)
            for pt_root_id_col, new_position_column in soma_position_mappings.items():
                synapses[f"{new_position_column}_voxels"] = synapses[pt_root_id_col].apply(lambda pt_root_id: soma_positions_voxels.get(pt_root_id, None))
                synapses[f"{new_position_column}_microns"] = synapses[pt_root_id_col].apply(lambda pt_root_id: soma_positions_microns.get(pt_root_id, None))

        return synapses


    def get_axonal_synapses(self, presyn_pt_root_id, microns_position_mappings=dict(pre_pt_position="synapse_position_microns"), **kwargs):
        return self._get_synapses(pre_ids=presyn_pt_root_id, microns_position_mappings=microns_position_mappings, **kwargs)


    def get_dendritic_synapses(self, postsyn_pt_root_id, microns_position_mappings=dict(post_pt_position="synapse_position_microns"), **kwargs):
        return self._get_synapses(post_ids=postsyn_pt_root_id, microns_position_mappings=microns_position_mappings, **kwargs)


    def _include_proofreading_inplace(self, table, flag, table_pt_root_id_key="pt_root_id", axon_proof_key="axon_proofreading", dendrite_proof_key="dendrite_proofreading"):
        def include_proof(proof, key):
            proof_status = pd.Series(index=proof.pt_root_id.values, data=proof.cell_type.values) # Map from pt_root_id --> proofreading status
            table[f"{key}_status"] = table[table_pt_root_id_key].apply(lambda root_id: proof_status.get(root_id, self.proofreading_status_none))
            table[f"{key}_complete"] = table[f"{key}_status"].isin(self.proofreading_status_complete)
        
        root_ids = table[table_pt_root_id_key].values

        if flag in ("axon", True, "both"):
            # Add column for axon proofreading status
            include_proof(self.get_axon_proofreading_table(root_ids=root_ids), axon_proof_key)
        
        if flag in ("dendrite", True, "both"):
            # Add column for dendrite proofreading status
            include_proof(self.get_dendrite_proofreading_table(root_ids=root_ids), dendrite_proof_key)

    def get_cell_type_table(self, drop_duplicates=True, transform_position=True, transform_position_column="position_microns", include_proofreading=True):
        cell_type_table = self.query_table("manual_central_types")

        if drop_duplicates:
            cell_type_table.drop_duplicates("pt_root_id", inplace=True)
            cell_type_table.reset_index(inplace=True)

        if transform_position:
            self.df_position_to_microns(cell_type_table, "pt_position", transform_position_column)

        self._include_proofreading_inplace(cell_type_table, flag=include_proofreading)
        
        return cell_type_table

    def _OLD_get_2p_corresponded_table(self, drop_duplicates=True, roi_column="roi", microns_position_mappings=dict(pt_position="position_microns"), include_proofreading=True):
        corresponded = self.query_table("correspondance_pilot")

        if drop_duplicates:
            corresponded.drop_duplicates("pt_root_id", inplace=True)
            corresponded.reset_index(inplace=True)

        if roi_column is not None:
            def get_corresponding_roi_by_row(row):
                cls_sys = row["classification_system"] # e.g., "session13"
                cell_type = row["cell_type"] # e.g., "plane2_roi_0269"
                col = int(cls_sys[-2])
                vol = cls_sys[-1]
                try: vol = int(vol)
                except: pass
                try:
                    if "," in cell_type: # Means the cell was recorded twice in 2P
                        # Choose the first one
                        cell_type = cell_type[:cell_type.find(",")]

                    split = cell_type.split("_")
                    if len(split) == 2:
                        plane, roi = split
                    elif len(split) == 3:
                        plane, _, roi = split
                    plane = int(plane[5:]) + 1 # IMPORTANT! The plane here is zero-indexed
                    roi = int(roi)
                    return f"M409828_{col}{vol}_{plane}_{roi}"
                except:
                    print("Bad cell type:", cell_type)
                    return None

            corresponded[roi_column] = corresponded.apply(get_corresponding_roi_by_row, axis=1)

        if microns_position_mappings is not None:
            for position_column, new_position_column in microns_position_mappings.items():
                self.df_position_to_microns(corresponded, position_column, new_position_column)

        self._include_proofreading_inplace(corresponded, flag=include_proofreading)

        return corresponded


    def get_coregistration_table(self, drop_duplicates=True, include_proofreading=True):
        coreg_table = self.query_table("manual_pilot_functional_coregistration_v1")
        coreg_table["roi"] = coreg_table.apply(lambda row: f"M409828_{row.session}{row.scan_idx}_{row.field+1}_{row.unit_id}", axis=1)
        
        self.df_position_to_microns(coreg_table, "pt_position", "position_microns")
        
        self._include_proofreading_inplace(coreg_table, flag=include_proofreading)
        
        return coreg_table



    def _get_proofreading_table(self, table, root_ids=None, drop_duplicates=True, only_complete=True):
        filter_in_dict = {}
        if root_ids is not None:
            filter_in_dict["pt_root_id"] = root_ids
        if only_complete:
            filter_in_dict["cell_type"] = self.proofreading_status_complete
        
        table = self.query_table(table, filter_in_dict=filter_in_dict)

        if drop_duplicates:
            table.drop_duplicates("pt_root_id", inplace=True)
            table.reset_index(inplace=True)
        
        table["complete"] = table["cell_type"].isin(self.proofreading_status_complete)

        return table


    def get_dendrite_proofreading_table(self, **kwargs):
        return self._get_proofreading_table(self.dendrite_proofreading_table, **kwargs)
    

    def get_axon_proofreading_table(self, **kwargs):
        return self._get_proofreading_table(self.axon_proofreading_table, **kwargs)


    # ==================================================
    # Tools for loading neuron skeletons/meshworks


    def _query_neuron_meshwork(self, root_id):
        root_point, root_point_res = self.get_single_soma_position(root_id, return_voxel_resolution=True)

        mw = coord_space_meshwork(
            root_id=root_id,
            client=self.cave_client,
            cv=self.cave_client_seg_cv,
            root_point=root_point,
            root_point_resolution=root_point_res,
            collapse_soma=True, # Collapse vertices within 7500 nm (collapse_radius) to a single soma
            synapses="all",
            synapse_table=self.synapse_table,
            # remove_self_synapse=True, # True by default
        )

        # Correct the synapse voxel resolutions
        for syn in ("pre_syn", "post_syn"):
            mw.anno[syn].voxel_resolution = self.get_voxel_res(mw.anno[syn].data_original)

        return mw
    
    def get_neuron_meshwork(self, root_id):
        if root_id in self._neuron_meshwork_cache:
            return self._neuron_meshwork_cache[root_id]
        else:
            mw = self._query_neuron_meshwork(root_id)
            self._neuron_meshwork_cache[root_id] = mw
            return mw

    def get_neurite_path_to_root(self, neuron_mw: Meshwork, skel_index: int=None, mesh_index: int=None):
        """Get the neurite path from a skeleton index to the neuron root (soma).

        Args:
            neuron_mw (Meshwork): Neuron Meshwork object
            skel_index (int): Start *skeleton* index of path
            mesh_index (int): Start *mesh* index of path

        Returns:
            np.ndarray: Positions on path, shape (n_steps_in_path, 3)
        """
        if mesh_index is not None:
            skel_index = neuron_mw._mind_to_skind(mesh_index)[0]
        path_skel_indices = neuron_mw.skeleton.path_to_root(skel_index)
        path_positions_nm = neuron_mw.skeleton.vertices[path_skel_indices] # Vertices are in nm units
        path_positions_microns = self.transform_position_to_microns(path_positions_nm)

        # NOTE: Path length can also be computed as the length of individual line segments in path_positions_microns
        # However, the V1DD transform does not involve any warping (only translation and rotation), so using path_length is fine.
        # Source: https://github.com/ceesem/standard_transform/blob/main/standard_transform/datasets.py#L18

        return path_positions_microns
    
    def get_neurite_distance_to_root(self, neuron_mw: Meshwork, skel_index=None, mesh_index: int=None):
        """Get the neurite path distance from a skeleton index to the neuron root (soma).

        Args:
            neuron_mw (Meshwork): Neuron Meshwork object
            skel_index (int or array): Start *skeleton* index of path
            mesh_index (int or array): Start *mesh* index of path

        Returns:
            float or array: Total length of path in microns (float if input is a single index, array if input is an array)
        """
        # NOTE: Path length can also be computed as the length of individual line segments in path_positions_microns above.
        # However, the V1DD transform does not involve any warping (only translation and rotation), so using path_length is fine.
        # Source: https://github.com/ceesem/standard_transform/blob/main/standard_transform/datasets.py#L18

        if mesh_index is not None:
            dist = neuron_mw.distance_to_root(mesh_indices=mesh_index)

            if type(mesh_index) is int:
                dist = dist[0]
            
            return dist / 1000
        elif skel_index is not None:
            return neuron_mw.skeleton.distance_to_root[skel_index] / 1000

        # if mesh_index is not None:
        #     return neuron_mw.distance_to_root(mesh_index)[0] / 1000
        # elif skel_index is not None:
        #     # path_length_microns = neuron_mw.skeleton.path_length(path_skel_indices) / 1000 # Another way of computing path length (same result)

        return None
    
    def plot_neuron_2d(self, neuron_mw: Meshwork, neuron_plot_dims=(0, 1), plot_type: str="mesh", color="black", alpha=0.25, point_size=1, highlight_soma: bool=True, ax=None, highlight_synapses=None, root_origin=None, dendritic_synapse_plotter=None, axonal_synapse_plotter=None, **kwargs):
        """Plot the skeleton or meshwork of a neuron on a 2D plot.

        Args:
            neuron_mw (Meshwork): Neuron Meshwork object
            neuron_plot_dims (tuple, optional): Dimensions of neuron to plot, formatted as (x_dimension, y_dimension). Defaults to (0, 1).
            plot_type (str, optional): Plotting type, either "mesh" or "skeleton". Defaults to "mesh".
            color (str, optional): Plotting color of neuron. Defaults to "black".
            alpha (float, optional): _description_. Defaults to 0.25.
            point_size (int, optional): _description_. Defaults to 1.
            highlight_soma (bool, optional): _description_. Defaults to True.
            highlight_synapses (any, optional): None or False for no synapses, "all" or True for all synapses, "dendritic" or "axonal" for dendritic or axonal synapses. Defaults to None.
            ax (_type_, optional): Matplotlib axis on which to plot. Defaults to None.
            root_origin (tuple, optional): Translate the plot so the root/soma is at the given coordinates. None means no translation, e.g., (100, None) puts the x-coordinate of the soma at 100. Defaults to None.
            dendritic_synapse_plotter (callable, optional): Args: ax, neuron_mw, synapse_points. Defaults to None.
            axonal_synapse_plotter (callable, optional): Args: ax, neuron_mw, synapse_points. Defaults to None.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.axis("equal")
        
        soma_position_microns = self.transform_position_to_microns(neuron_mw.skeleton.root_position)[list(neuron_plot_dims)]
        offset = [0 if (root_origin is None or root_origin[i] is None) else (root_origin[i] - soma_position_microns[i]) for i in range(2)]
        
        vertices = neuron_mw.skeleton.vertices if plot_type == "skeleton" else neuron_mw.mesh.vertices
        transformed_vertices = self.transform_position_to_microns(vertices)[:, neuron_plot_dims] + offset
        ax.scatter(transformed_vertices[:, 0], transformed_vertices[:, 1], s=point_size, color=color, alpha=alpha, **kwargs)

        if highlight_soma:
            soma_vertices = self.transform_position_to_microns(neuron_mw.mesh.vertices[neuron_mw.root_region])[:, neuron_plot_dims] + offset
            ax.scatter(soma_vertices[:, 0], soma_vertices[:, 1], color="black", s=10, alpha=1)

        syn_size = 0.5
        syn_alpha = 0.5

        if highlight_synapses in (True, "all", "a", "axo", "axonal"):
            syn_pts = self.transform_position_to_microns(neuron_mw.anno.pre_syn.points)[:, neuron_plot_dims] + offset
            if axonal_synapse_plotter is not None:
                axonal_synapse_plotter(ax, neuron_mw, syn_pts)
            else:
                ax.scatter(syn_pts[:, 0], syn_pts[:, 1], s=syn_size, color="tomato", alpha=syn_alpha)
        if highlight_synapses in (True, "all", "d", "den", "dendritic"):
            syn_pts = self.transform_position_to_microns(neuron_mw.anno.post_syn.points)[:, neuron_plot_dims] + offset
            if dendritic_synapse_plotter is not None:
                dendritic_synapse_plotter(ax, neuron_mw, syn_pts)
            else:
                ax.scatter(syn_pts[:, 0], syn_pts[:, 1], s=syn_size, color="turquoise", alpha=syn_alpha)

    
    # def dist_on_path(path_positions):
    #     dist = 0
    #     for i in range(len(path_positions) - 1):
    #         dist += np.linalg.norm(path_positions[i+1] - path_positions[i])
    #     return dist