
import networkx as nx
import numpy as np
from walkers.walker import Walker


class DeepWalker(Walker):
    def __init__(self, graph, workers=1, dimensions=64, walk_len=10, num_walks=200):
        print("Deep Walker Sampling Method")
        super().__init__(graph, workers=workers, dimensions=dimensions, walk_len=walk_len, num_walks=num_walks)
       
        self.number_of_nodes = self.graph.number_of_nodes()

        # Transition Prs matrix
        self.d_graph = {node: {"pr":list(), "ngh":list()} for node in self.graph.nodes()}

        # compute probabilities
        print("!!!!  Computing Probability Matrix")
        self._precompute_probabilities()
        print("!!!!  Generate Walks")
        self._generate_walks(self.graph, self.d_graph, type="local")


    def _precompute_probabilities(self):
        for i in self.graph.nodes():
            # we traverse neighbours only because for non neighbours this value should be zero
            # according to formula 
            neighbors = list(self.graph.successors(i))
       
            if len(neighbors) != 0: # denominator is non zero
                _sum = 0.0
                for (_,j) in np.ndenumerate(neighbors):                
                    w = self.graph[i][j].get(self.weight_key, 1)
                    num = w
                    _sum += num
                    self.d_graph[i]["pr"].append(num)
                    self.d_graph[i]["ngh"].append(j)   

                self.d_graph[i]["pr"] =  np.array(self.d_graph[i]["pr"])/_sum     


 
