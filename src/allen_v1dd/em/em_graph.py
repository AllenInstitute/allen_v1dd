import networkx as nx
from networkx.readwrite.gml import read_gml, write_gml

class EMGraph():
    def __init__(self, filename=None, read_from_file=False, nx_graph_class=nx.MultiDiGraph):
        self.filename = filename

        if read_from_file:
            self.graph = read_gml(filename)
        else:
            self.graph = nx_graph_class()

    def save(self):
        """Saves the graph to the save file, if specified.
        """
        if self.filename is not None:
            write_gml(self.graph, self.filename)
    
