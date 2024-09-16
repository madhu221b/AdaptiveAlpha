
import networkx as nx
import numpy as np

try:
  from .walker import Walker
except Exception as error:
    from walker import Walker

class InDegreeWalker(Walker):
    def __init__(self,graph,beta=0,workers=1,dimensions=64,walk_len=10,num_walks=200):
        print("In Degree Walker with beta = ", beta)
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)
        """pi i-> j =  A_ij (q_j_indegree^beta) / sum l=1 ^ N Ail q_l_indegree^beta """
        self.number_of_nodes = self.graph.number_of_nodes()

        # Transition Prs matrix
        # self.pi = np.zeros((self.number_of_nodes, self.number_of_nodes))
        self.d_graph = {node: {"pr":list(), "ngh":list()} for node in self.graph.nodes()}
    
        degree = dict(self.graph.in_degree()) # note now it is indegree
        self.degree_pow = dict({node: (np.round(degree**beta,5) if degree != 0 else 0) for node, degree in degree.items()})
    
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
                    deg = self.degree_pow[j]
                    w = self.graph[i][j].get(self.weight_key, 1)
                    num = deg*w
                    _sum += num
                    self.d_graph[i]["pr"].append(num)
                    self.d_graph[i]["ngh"].append(j)   

                self.d_graph[i]["pr"] =  self.d_graph[i]["pr"]/_sum     


 
