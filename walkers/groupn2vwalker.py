from collections import Counter
import networkx as nx
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import copy

import torch
import dgl
from dgl.sampling import random_walk

try:
  from .walker import Walker
except Exception as error:
    from walker import Walker

class GroupN2VWalk(Walker):
    def __init__(self, graph, workers=1, dimensions=128, walk_len=20, walks_per_node=10, pq_dict=dict()):
        print(f"Custom N2V Walker group specific p and q: {pq_dict}")
        super().__init__(graph, workers=workers, dimensions=dimensions, walk_len=walk_len, walks_per_node=walks_per_node)
        self.quiet = False
        self.is_optimise = False # For huge graphs, relying less on networkx
        self.number_of_nodes = self.graph.number_of_nodes()
        self.device = "cpu"
        self.pq_dict = pq_dict
        self.node_attrs = nx.get_node_attributes(self.graph, "group")

        self.preprocess_transition_probs()
        self.walks = self.simulate_walks()
        

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.graph
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
                        alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self):
        '''
        Repeatedly simulate random walks from each node.
        '''
        walks = []
        nodes = list(self.graph.nodes())
        
        for walk_iter in range(self.walks_per_node):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=self.walk_len, start_node=node))
        
        walks = [[str(node) for node in walk] for walk in walks]
        print(f"Walk results shape:({len(walks)},{len(walks[0])}) ", )
        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        p = self.pq_dict[self.node_attrs[src]]["p"]
        q = self.pq_dict[self.node_attrs[src]]["q"]


        unnormalized_probs = []
        for dst_nbr in sorted(self.graph.neighbors(dst)):
            weight = 1 # unweighted graph
            if dst_nbr == src:
                unnormalized_probs.append(weight/p)
            elif self.graph.has_edge(dst_nbr, src):
                unnormalized_probs.append(weight)
            else:
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        alias_nodes = {}
        for node in self.graph.nodes():
            unnormalized_probs = [1 for nbr in sorted(self.graph.neighbors(node))] # unweighted graphs
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}


        for edge in self.graph.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])


        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges



def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int32)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]