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
  from .fastdeepwalk import FastDeepWalker
except Exception as error:
    from walker import Walker
    from fastdeepwalk import FastDeepWalker
   
# from .walkers.fastdeepwalk import DeepWalker

class FastCrossWalk(Walker):
    def __init__(self, graph, beta=0,  p=1, alpha=0.5, workers=1, dimensions=64, walk_len=10, num_walks=200):
        print(" FAST Cross Walker with p: {}, alpha: {} (Implementation only for 2 groups)".format(p, alpha))
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)
        self.p , self.alpha = p, alpha
        self.quiet = False
        self.number_of_nodes = self.graph.number_of_nodes()
        
        # self.device =  'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = "cpu"
        self.edge_dict = dict()
      
        self.node_attrs = nx.get_node_attributes(self.graph, "group")
        self.group_df = pd.DataFrame.from_dict(self.node_attrs, orient='index', columns=['group'])
        self.groups = set(self.node_attrs.values())

    
        print("Computing node embeddings on {}".format(self.device))
 
        print("!! Precomputing Probablities -  Fast Crosswalker")
        print("!!!!  1. Generate Walks on Deep Walks")
        deepwalker = FastDeepWalker(graph)
        walks = deepwalker.walks
        walks = walks.to(self.device)
        print("!!!!  2. Compute Weights")
        u, v, w = self._get_weight(walks)
        print(u.size(), v.size(), w.size())
        print("!!! Create Graph from these weights")
        self._precompute_graph(u, v, w)

        print("!! Precomputing Probablities -  Fast Crosswalk")
        self._precompute_probabilities() # populate d_graph
        print("!!!!  Generate Walks")
        self.walks = self._generate_walks()
        

        
    def _precompute_graph(self, u, v, w):
        edges = u, v
        weights = w # weight of each edge
        self.dgl_g = dgl.graph(edges).to(self.device)
        self.dgl_g.edata['w'] = weights  # give it a name 'w'

        self.dgl_g.edata['p'] = torch.zeros(self.dgl_g.num_edges(), dtype=torch.float32, device=self.device)

                
    def _get_weight(self, walks):
        ##  Calculating closeness to boundary
        r, d = self.num_walks, self.walk_len
        start_nodes = walks[:, 0]

        walks_id = torch.tensor(walks).apply_(lambda x: self.node_attrs[int(x)])
        start_nodes_id = walks_id[:, 0].unsqueeze(1).repeat(1,d)
        
        walks_id = walks_id[:, 1:]
        not_eq_ids = ~torch.eq(walks_id, start_nodes_id)
        sum_not_eq_ids = torch.sum(not_eq_ids, dim=1)
        
        cat = torch.vstack((start_nodes, sum_not_eq_ids)).T
        df = pd.DataFrame(cat.numpy(), columns=["start_node","notsameid"])
        groupby = df.groupby(['start_node'])
        groupby = groupby['notsameid'].sum()/(r*d)
        groupby = groupby.to_frame()
        groupby['m_v'] = groupby["notsameid"] ** self.p
   

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
                    w_vu = Nv["m_v"]*(1-self.alpha)/Z
                    list_ws.extend(w_vu)
                else:
                    list_ws.extend([1]*Nv.shape[0])

           
             # Edges Connecting Different Groups
            if Rv.shape[0] != 0:
                Z = Rv.shape[0]*Rv["m_v"].sum()
                list_us.extend([v]*Rv.shape[0])
                list_vs.extend(list(Rv.index))
                if Nv.shape[0] != 0: # Crosswalk's condition
                    if Z: 
                       w_vu = Rv["m_v"]*(self.alpha)/Z
                       list_ws.extend(w_vu)
                    else:
                       list_ws.extend([1]*Rv.shape[0]) 
                else:
                    if Z: 
                       w_vu = Rv["m_v"]/Z
                       list_ws.extend(w_vu)
                    else:
                       list_ws.extend([1]*Rv.shape[0]) 

        
            
            
        return torch.tensor(list_us).to(torch.int64), torch.tensor(list_vs).to(torch.int64), torch.tensor(list_ws)

           
    def _precompute_probabilities(self):
        # Data structures to populate pr
        # device = "cpu" if self.is_optimise else self.device
        device = "cpu"
        us, vs, ps = list(), list(), list()
        print("initializing edges for torch pr", self.dgl_g.num_edges())
        torch_prs = torch.zeros(self.dgl_g.num_edges(), dtype=torch.float32, device=device)
        
        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')
        
        for u in nodes_generator:
            local_successors = list(self.graph.successors(u))
            len_local = len(local_successors)
            if len_local == 0: continue 

            list_us = torch.tensor(u).to(torch.int64).to(device).repeat(len_local)
            list_vs = torch.tensor(local_successors).to(device)
            edge_ids = self.dgl_g.edge_ids(list_us,list_vs)

            ws = self.dgl_g.edata['w'][edge_ids]
            sum_ws = torch.sum(ws)
            if sum_ws == 0: continue
            else: list_prs = ws/sum_ws
            torch_prs[edge_ids] = list_prs

        self.dgl_g.edata['p'] = torch_prs
    
            

    def _generate_walks(self) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """
        # Split num_walks for each worker
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Generating walks on : ", self.device)
        nodes = torch.tensor(list(self.graph.nodes()))
        nodes = nodes.repeat(self.num_walks).to(self.device)

        walk_results, _ = dgl.sampling.random_walk(self.dgl_g.to(self.device), nodes=nodes, length=self.walk_len, prob="p")
        print("Walk results shape: ", walk_results.size())
        # walks = walk_results
        walks = np.array(walk_results.tolist()).astype(str).tolist()
        # walks = np.array(walk_results.tolist()).tolist()
        return walks

