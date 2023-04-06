from collections.abc import Collection

import pandas as pd

from ..client import EMClient
from . import EMGraph

class DynamicSynapseGraph(EMGraph):
    SYN_ATTRIBUTES = {
        "size": int,
        "soma_soma_dist": float,
        "soma_soma_dist_horiz": float
    }

    def __init__(self, em_client: EMClient, debug=True, **kwargs):
        super().__init__(**kwargs)
        self.em_client = em_client
        self.debug = debug


    def _has_loaded_syn(self, id, syn_type):
        id = int(id)
        return self.graph.has_node(id) and self.graph.nodes[id].get(f"{syn_type}_syn_loaded") == True
    

    def _set_node_attr(self, ids: list, attr: str, value: any):
        for id in ids:
            id = int(id)
            # if not self.graph.has_node(id):
            #     self.graph.add_node(id, {attr: value})
            self.graph.nodes[id][attr] = value


    def _update_graph_with_synapses(self, synapse_df, syn_type):
        # Make sure we don't doubly add synapse edges
        # synapse_df is a ground truth table for pre --> post synapses
        n_syn_before = len(synapse_df)
        if syn_type == "axo":
            # If they are axonal synapses, then don't add an edge from A to B if B already has loaded inputs (dendrite)
            post_ids = synapse_df["post_pt_root_id"].unique()
            post_ids_with_loaded_dendrites = [id for id in post_ids if self._has_loaded_syn(id, "den")]
            if len(post_ids_with_loaded_dendrites) > 0:
                synapse_df = synapse_df[~synapse_df.post_pt_root_id.isin(post_ids_with_loaded_dendrites)]
        elif syn_type == "den":
            # If they are dendritic synapses, then don't add an edge from A to B if A already has loaded outputs (axon)
            pre_ids = synapse_df["pre_pt_root_id"].unique()
            pre_ids_with_loaded_axons = [id for id in pre_ids if self._has_loaded_syn(id, "axo")]
            if len(pre_ids_with_loaded_axons) > 0:
                synapse_df = synapse_df[~synapse_df.pre_pt_root_id.isin(pre_ids_with_loaded_axons)]
        n_syn_after = len(synapse_df)

        if self.debug and n_syn_after < n_syn_before:
            print(f"Ignoring {n_syn_before-n_syn_after} synapses because of cells with already-loaded axonal or dendritic synapses")

        # Add edges
        for _, row in synapse_df.iterrows():
            pre_id = int(row["pre_pt_root_id"])
            post_id = int(row["post_pt_root_id"])
            attrs = {attr: attr_type(row[attr]) for attr, attr_type in self.SYN_ATTRIBUTES.items()}
            self.graph.add_edge(pre_id, post_id, **attrs)


    def get_synapses(self, root_ids, syn_type) -> pd.DataFrame:
        """Loads synapses of a certain type (axonal or dendritic) for a given set of neurons.
        This method will query for synapses if they have not already been loaded in the graph.

        Args:
            root_ids (int or list): List of root IDs
            syn_type (str): Type of synapse, "axonal" or "dendritic"
        
        Returns: 
            pd.DataFrame: Synapse table with columns
                - pre_pt_root_id: Presynaptic root id
                - post_pt_root_id: Postsynaptic root id
                - size: Synapse volume
                - soma_soma_dist: Euclidean distance between pre and post soma, or inf if one soma is out of the EM volume.
                - soma_soma_dist_horiz: Horizontal distance between pre and post soma, or inf if one soma is out of the EM volume.
        """
        if not isinstance(root_ids, Collection): root_ids = [root_ids]

        if syn_type in ("a", "axo", "axonal"):
            syn_type = "axo"
        elif syn_type in ("d", "den", "dendritic"):
            syn_type = "den"
        else:
            raise ValueError(f"bad syn_type: '{syn_type}'")

        ids_to_load = [id for id in root_ids if not self._has_loaded_syn(id, syn_type)]

        if len(ids_to_load) > 0:
            if self.debug: print(f"Loading {syn_type} synapses for {len(ids_to_load)} neurons.")

            # Load synapses (expensive query)
            if syn_type == "axo":
                synapse_df = self.em_client.get_axonal_synapses(root_ids)
            elif syn_type == "den":
                synapse_df = self.em_client.get_dendritic_synapses(root_ids)

            self._update_graph_with_synapses(synapse_df, syn_type) # Update graph with loaded synapses
            self._set_node_attr(ids_to_load, f"{syn_type}_syn_loaded", True) # Mark nodes as having loaded synapses
            self.save() # Save graph to file
        
        # Now that we know the graph is updated with the synapses, query for the edges
        df = []
        for id in root_ids:
            if syn_type == "axo":
                edge_iter = self.graph.out_edges(id, data=True)
            elif syn_type == "den":
                edge_iter = self.graph.in_edges(id, data=True)

            for pre, post, data in edge_iter:
                row = dict(pre_pt_root_id=pre, post_pt_root_id=post)
                row.update(data)
                df.append(row)
        
        df = pd.DataFrame(df)
        return df

    def get_axonal_synapses(self, pre_ids) -> pd.DataFrame:
        """Loads all axonal synapses for a given set of neuruons.
        (i.e., all synapses where the pre_root_id is in pre_ids.)

        Args:
            pre_ids (int or list): Presynaptic root ids

        Returns:
            pd.DataFrame: Synapse table (see get_synapses method for details)
        """
        return self.get_synapses(pre_ids, syn_type="axo")

    def get_dendritic_synapses(self, post_ids) -> pd.DataFrame:
        """Loads all dendritic synapses for a given set of neuruons.
        (i.e., all synapses where the post_root_id is in post_ids.)

        Args:
            post_ids (int or list): Postsynaptic root ids

        Returns:
            pd.DataFrame: Synapse table (see get_synapses method for details)
        """
        return self.get_synapses(post_ids, syn_type="den")