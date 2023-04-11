from os import path

import networkx as nx
from networkx.readwrite.gml import read_gml, write_gml

class EMGraph():
    def __init__(self, filename=None, nx_graph_class=nx.MultiDiGraph):
        self.filename = filename

        graph = None
        if filename is not None:
            print(f"Loading graph from file {filename}")
            if path.isfile(filename):
                graph = read_gml(filename, destringizer=int) # because unfortunately node labels are stored as strings (only support for 32-bit ints)
        
        self.graph = nx_graph_class() if graph is None else graph

    def save(self):
        """Saves the graph to the save file, if specified.
        """
        if self.filename is not None:
            write_gml(self.graph, self.filename)
    
    def describe_memory_usage(self):
        import sys
        edge_mem = sum([sys.getsizeof(e) for e in self.graph.edges(data=True)])
        node_mem = sum([sys.getsizeof(n) for n in self.graph.nodes(data=True)])
        print("Edge memory:", edge_mem)
        print("Node memory:", node_mem)
        print("Total memory:", (edge_mem + node_mem) / 1e9, "GB")

    def describe_size(self):
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print(f"|V| = {n_nodes}, |E| = {n_edges}")