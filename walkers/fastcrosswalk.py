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
  from .n2vwalker import N2VWalk
  from .n2vwalker import alias_setup, alias_draw

except Exception as error:
    from walker import Walker
    from n2vwalker import N2VWalk
    from n2vwalker import alias_setup, alias_dra
   
# from .walkers.fastdeepwalk import DeepWalker

class FastCrossWalk(Walker):
    def __init__(self, graph, p_cw = 1, alpha_cw = 0.5, workers=1, dimensions=128, walk_len=20, walks_per_node=10):
        print(f"Fast Cross Walker with p_cw: {p_cw}, alpha: {alpha_cw}")
        super().__init__(graph, workers=workers, dimensions=dimensions, walk_len=walk_len, walks_per_node=walks_per_node)
        self.p_cw , self.alpha_cw = p_cw, alpha_cw
        self.p, self.q = 1, 1 
        self.quiet = False
        self.number_of_nodes = self.graph.number_of_nodes()
        

        self.device = "cpu"

      
        self.node_attrs = nx.get_node_attributes(self.graph, "group")
        self.group_df = pd.DataFrame.from_dict(self.node_attrs, orient='index', columns=['group'])
        self.groups = set(self.node_attrs.values())

    
        print("Computing node embeddings on {}".format(self.device))
 
        print("!! Precomputing Probablities -  Fast Crosswalker")
        print("!!!!  1. Generate Walks using Node2Vec")
        n2vwalker = N2VWalk(graph)
        walks = n2vwalker.walks
        walks = [list(map(int, row)) for row in walks]
        print("!!!!  2. Compute Weights")
        weight_dict = self._get_weight(walks)
        print("!!! Create Graph from these weights")
        self._precompute_graph(weight_dict)

        print("!! Precomputing Probabilities -  Fast Crosswalk")
        self.precompute_probabilities() # populate d_graph
        print("!!!!  Generate Walks")
        self.walks = self.simulate_walks()
        
   

    def multidigraph_to_digraph(self, multi_g):
        g = nx.DiGraph()
        g.add_nodes_from(multi_g.nodes(data=True))

        for u, v, data in multi_g.edges(data=True):
            weight = data.get("weight", 1)  # default to 1 if no weight
            if g.has_edge(u, v):
                g[u][v]["weight"] += weight  # sum weights
            else:
                g.add_edge(u, v, weight=weight)
        
        return g
        
    def _precompute_graph(self, weight_dict):
        self.graph = self.multidigraph_to_digraph(self.graph)
        nx.set_edge_attributes(self.graph, weight_dict)

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(self.graph.neighbors(dst)):
            weight = self.graph[dst][dst_nbr]['weight'] # weighted graph
            if dst_nbr == src:
                unnormalized_probs.append(weight/p)
            elif self.graph.has_edge(dst_nbr, src):
                unnormalized_probs.append(weight)
            else:
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def precompute_probabilities(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        alias_nodes = {}
        for node in self.graph.nodes():
            unnormalized_probs = [self.graph[node][nbr]["weight"] for nbr in sorted(self.graph.neighbors(node))] # weighted graphs
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}


        for edge in self.graph.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])


        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
    
    
                
    def _get_weight(self, walks):
        ##  Calculating closeness to boundary
        r, d = self.walks_per_node, self.walk_len
        start_nodes = [walk[0] for walk in walks]

        walks_id = [[self.node_attrs[node] for node in walk] for walk in walks]
        start_nodes_id = [[self.node_attrs[start_node]]*(len(walk)-1) for walk, start_node in zip(walks, start_nodes)]
        
        walks_id = [walk_id[1:] for walk_id in walks_id]
        sum_not_eq_ids = [torch.sum(~torch.eq(torch.tensor(walk_id), torch.tensor(start_node_id))) for walk_id, start_node_id in zip(walks_id, start_nodes_id)]
        
        cat = torch.vstack((torch.tensor(start_nodes), torch.tensor(sum_not_eq_ids))).T
        df = pd.DataFrame(cat.numpy(), columns=["start_node","notsameid"])
        groupby = df.groupby(['start_node'])
        groupby = groupby['notsameid'].sum()/(r*d)
        groupby = groupby.to_frame()
        groupby['m_v'] = groupby["notsameid"] ** self.p_cw
   

        ## Some pandas operations to have m_v and node id in same dataframe
        groupdf = self.group_df.rename_axis('start_node').reset_index()
        join_df = pd.concat([groupdf,groupby], axis=1).fillna(0)
        
        ## Reweighting Edges
        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing reweighting of edges')
        
        list_us, list_vs, list_ws = list(), list(), list()
        for v in nodes_generator:
            successors = list(self.graph.successors(v))
            if len(successors) == 0: continue
            succ_df = join_df.loc[successors,:]
            id_v = join_df.loc[v, "group"]
            Nv = succ_df.loc[succ_df["group"] == id_v, :]
            Rv = succ_df.loc[succ_df["group"] != id_v, :]

            # Edges in Same group
            if Nv.shape[0] != 0:
                Z = Nv['m_v'].sum()
                list_us.extend([v]*Nv.shape[0])
                list_vs.extend(list(Nv.index))
                if Z: 
                    w_vu = ((1-self.alpha_cw)*Nv["m_v"])/Z
                    list_ws.extend(w_vu)
                else:
                    list_ws.extend([0.01]*Nv.shape[0])

           
             # Edges Connecting Different Groups
            if Rv.shape[0] != 0:
                Z = Rv.shape[0]*Rv["m_v"].sum()
                list_us.extend([v]*Rv.shape[0])
                list_vs.extend(list(Rv.index))
                if Nv.shape[0] != 0: # Crosswalk's condition
                    if Z: 
                       w_vu = (self.alpha_cw*Rv["m_v"])/Z
                       list_ws.extend(w_vu)
                    else:
                       list_ws.extend([0.01]*Rv.shape[0]) 
                else:
                    if Z: 
                       w_vu = Rv["m_v"]/Z
                       list_ws.extend(w_vu)
                    else:
                       list_ws.extend([0.01]*Rv.shape[0]) 

        weight_dict = {(u,v):{"weight":w} for u, v, w in zip(list_us, list_vs, list_ws)}
        return weight_dict

           
    def cw_walk(self, walk_length, start_node):
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
                walks.append(self.cw_walk(walk_length=self.walk_len, start_node=node))
        
        walks = [[str(node) for node in walk] for walk in walks]
        print(f"Walk results shape:({len(walks)},{len(walks[0])}) ", )
        return walks
    
            


