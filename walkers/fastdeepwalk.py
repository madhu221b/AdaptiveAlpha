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

class FastDeepWalker(Walker):
    def __init__(self, graph, workers=1, dimensions=64, walk_len=10, num_walks=200):
        print(" FAST Deep Walk")
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)
        self.quiet = False
        self.number_of_nodes = self.graph.number_of_nodes()
        
        # self.device =  'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = "cpu"

        print("Computing node embeddings on {}".format(self.device))
        print("!! Obtain DGL Graph from Networkx")
        self._precompute_graph() # convert to dgl graph
        print("!! Precomputing Probablities -  Deepwalk")
        self._precompute_probabilities() # populate d_graph
        print("!!!!  Generate Walks")
        self.walks = self._generate_walks()
    

    def _precompute_graph(self):

        self.clone_graph = self.graph.copy()
        self.dgl_g = dgl.from_networkx(self.clone_graph) 
        self.dgl_g = self.dgl_g.to(self.device)
        self.dgl_g.edata['p'] = torch.zeros(self.dgl_g.num_edges(), dtype=torch.float32, device=self.device)

                
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

            # assign local-weight
            if len_local != 0: local_pr = 1/len_local
            

            pr = torch.tensor([local_pr], device=device)
            local_pr = pr.repeat(len_local)
   
            list_us = torch.tensor(u).to(torch.int64).to(device).repeat(len_local)
            list_vs = torch.tensor(local_successors)
            list_vs = list_vs.to(torch.int64).to(device)
            list_prs = local_pr
            
            edge_ids = self.dgl_g.edge_ids(list_us,list_vs)
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
        walks = walk_results
        # dgl adds "-1" if walks cannot be done , filter those rows
        condition = walks > 0
        row_cond = condition.all(1)
        walks = walks[row_cond, :] 
        # walks = np.array(walk_results.tolist()).astype(str).tolist()
        # walks = np.array(walk_results.tolist()).tolist()
        return walks

