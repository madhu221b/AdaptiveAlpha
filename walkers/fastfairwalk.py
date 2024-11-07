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

class FastFairWalk(Walker):
    def __init__(self, graph, workers=1, dimensions=64, walk_len=10, num_walks=200, p=1, q=1):
        print(" Fast Fairwalk Walker - with assumed p=1, q=1")
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)
        self.quiet = False
        self.is_optimise = False # For huge graphs, relying less on networkx
        self.number_of_nodes = self.graph.number_of_nodes()
        
        self.device = "cpu"
        self.edge_dict = dict()
      
        self.node_attrs = nx.get_node_attributes(self.graph, "group")
        self.group_df = pd.DataFrame.from_dict(self.node_attrs, orient='index', columns=['group'])
        self.groups = set(self.node_attrs.values())

    
        print("Computing node embeddings on {}".format(self.device))
        print("!! Obtain DGL Graph from Networkx")
        self._precompute_graph() # convert to dgl graph
        print("!! Precomputing Probablities -  Adaptive Alpha - NonLocal ID + Local RW (p=1,q=1)")
        self._precompute_probabilities() # populate d_graph
        print("!!!!  Generate Walks")
        self.walks = self._generate_walks()
    
        
    def _precompute_graph(self):
        self.dgl_g = dgl.from_networkx(self.graph, node_attrs=['group'])
        self.dgl_g.edata['p'] = torch.zeros(self.dgl_g.num_edges(), dtype=torch.float32)
        self.dgl_g = self.dgl_g.to(self.device)


    def _precompute_probabilities(self):
        # Data structures to populate pr
        # device = "cpu" if self.is_optimise else self.device
        device = "cpu"
        us, vs, ps = list(), list(), list()
        print("initializing edges for torch pr", self.dgl_g.num_edges())
        torch_prs = torch.zeros(self.dgl_g.num_edges(), dtype=torch.float32, device=device)
        
        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

     
                    
            # list_vs = torch.cat((torch.tensor(local_successors),torch.tensor(non_local_successors)))
            # list_vs = list_vs.to(torch.int64)
            # list_prs = torch.cat((local_pr,non_local_pr))
        for u in nodes_generator:
            successors = list(self.graph.successors(u))
            if len(successors) == 0: continue
            succ_df = self.group_df.loc[successors, :]

            
            count_series =  succ_df.groupby("group")["group"].count()
            count_df = pd.DataFrame({'group':count_series.index, 'count':count_series.values})
            count_df.set_index('group')
            count_df["pr"] = 1/count_df["count"]

            n_groups = count_df.shape[0]
            group_pr = 1/n_groups
            
            join_df = pd.merge(succ_df,count_df, left_on='group', right_on='group', how='left').drop(['count'],axis=1)
            join_df.set_index(succ_df.index,inplace=True)
            join_df["total_pr"] = join_df["pr"]*group_pr 

   
            list_us = torch.tensor(u).to(device).repeat(len(successors))
            list_vs = torch.tensor(successors).to(device)
            list_prs = torch.tensor(list(join_df.loc[successors, "total_pr"])).to(device)
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
        # walks = walk_results
        walks = np.array(walk_results.tolist()).astype(str).tolist()
        # walks = np.array(walk_results.tolist()).tolist()
        return walks

