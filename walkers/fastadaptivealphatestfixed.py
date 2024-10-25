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

class FastAdaptiveAlphaTestFixed(Walker):
    def __init__(self, graph, beta=0, alpha=0, workers=1, dimensions=64, walk_len=10, num_walks=200):
        print(" FAST Test Adaptive Alpha Non Local In Degree Walker with constant beta: {}  and alpha: {} And Local Random Walker ".format(beta,alpha))
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)
        self.quiet = False
        self.number_of_nodes = self.graph.number_of_nodes()
        
        self.device =  'cuda' if torch.cuda.is_available() else 'cpu'
        self.edge_dict = dict()
        self.alpha = alpha
        self.node_attrs = nx.get_node_attributes(self.graph, "group")
        self.group_df = pd.DataFrame.from_dict(self.node_attrs, orient='index', columns=['group'])
        self.groups = set(self.node_attrs.values())
 
        # Populate nodes by group
        self._get_group_to_node_dict()
            
        self.walk_types = ["nonlocal","local"]  

        degree = dict(self.graph.in_degree()) # note now it is indegree
        self.indegree_df = pd.DataFrame.from_dict(degree, orient='index', columns=['degree'])
        degree_pow = dict({node: (np.round(degree**beta,5) if degree != 0 else 0) for node, degree in degree.items()})
        self.degree_pow_df = pd.DataFrame.from_dict(degree_pow, orient='index', columns=['degree_pow'])
      
    
        print("Computing node embeddings on {}".format(self.device))
        print("!! Obtain DGL Graph from Networkx")
        self._precompute_graph() # convert to dgl graph
        print("!! Precomputing Probablities -  Adaptive Alpha - NonLocal ID + Local RW (p=1,q=1)")
        self._precompute_probabilities() # populate d_graph
        print("!!!!  Generate Walks")
        self.walks = self._generate_walks()
    
    def _get_group_to_node_dict(self):
        self.group_to_node_dict = dict()
        for node, node_id in self.node_attrs.items():
            if node_id not in self.group_to_node_dict:
                self.group_to_node_dict[node_id] = list()
            self.group_to_node_dict[node_id].append(node)

    def _get_non_local_successors(self, node, successors):
        non_local_jump_nodes = list()
        id_u = self.node_attrs[node]

        next_succ = [int(v) for successor in successors for v in list(self.dgl_g.successors(successor))]
        non_local_jump_nodes = self.group_df.loc[(self.group_df.index.isin(next_succ)) & ~(self.group_df.index.isin(successors)) & (self.group_df.group == id_u), :].index
         
        if len(non_local_jump_nodes) != 0:
            all_nodes = non_local_jump_nodes
        else: 
            all_nodes = self.group_to_node_dict[self.node_attrs[node]]
            all_nodes = list(set(all_nodes) - set(set(successors) | set([node])))
    
        return all_nodes
    
    def _get_non_local_successors_v2(self, node, successors):
        non_local_jump_nodes = list()
        for successor in successors:
            next_succ = self.graph.successors(successor)
            # not already connected to node or is an exisiting successor and is so same identity
            next_succ = [_ for _ in next_succ if _ != node and _ not in successors and self.node_attrs[_]==self.node_attrs[node]]
            non_local_jump_nodes.extend(next_succ)
        
        if len(non_local_jump_nodes) != 0:
            all_nodes = list(set(non_local_jump_nodes))
        else: 
            all_nodes = self.group_to_node_dict[self.node_attrs[node]]
            all_nodes = list(set(all_nodes) - set(set(successors) | set([node])))
    
        return all_nodes

    def _precompute_graph(self):
        # find all non-local edges
        print("find non-local edges in networkx")
        edge_list = list()
        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing non-local edges')
        for u in nodes_generator:
            successors = list(self.graph.successors(u))
            non_local_successors = self._get_non_local_successors_v2(u, successors)
            self.edge_dict[u] = {"local":successors, "nonlocal":non_local_successors}
            edge_list.extend([(u,v) for v in non_local_successors])
        
        # set edge attribute value for local edges
        nx.set_edge_attributes(self.graph, 1, "is_local")
        self.clone_graph = self.graph.copy()
        self.clone_graph.add_edges_from(edge_list, is_local=0)
        self.dgl_g = dgl.from_networkx(self.clone_graph, node_attrs=['group'], edge_attrs=["is_local"])
        self.dgl_g.edata['p'] = torch.zeros(self.dgl_g.num_edges(), dtype=torch.float32)
        self.dgl_g = self.dgl_g.to(self.device)
        
 


    def _precompute_probabilities(self):
        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')
        us, vs, ps = list(), list(), list()
        torch_prs = torch.zeros(self.dgl_g.num_edges(), dtype=torch.float32, device=self.device)
        for u in nodes_generator:
            all_successors = list(self.dgl_g.successors(u))
            local_successors, non_local_successors = self.edge_dict[u]["local"], self.edge_dict[u]["nonlocal"]
            assert len(all_successors) == len(local_successors) + len(non_local_successors)
            len_all = len(local_successors)
            # calculate alpha
            id_u = self.node_attrs[u]
            len_v = len([_ for _ in local_successors if self.node_attrs[_] != id_u])
            # print(len_v, len_all)
            if len_all == 0: alpha = 1.0
            else: alpha = self.alpha # alpha is fixed
            
            # assign local-weight
            if len_all != 0:
                 local_pr = 1/len_all
            else:
                local_pr = 0

            one_minus_alpha  = (1-alpha)
            pr = torch.tensor([one_minus_alpha*local_pr], device=self.device)
            # list_us = torch.tensor(u).to(self.device).repeat(len_all)
            # local_edge_ids = self.dgl_g.edge_ids(list_us, torch.tensor(local_successors).to(self.device))
            local_pr = pr.repeat(len_all)
            # torch_prs[local_edge_ids] = local_pr
       

            # assign non-local weight
            
            degree_df = self.degree_pow_df.loc[non_local_successors, :]
            sum_dfs = degree_df['degree_pow'].sum()
            # normalize the column and multiply by alpha
            degree_df["degree_pow"] = (alpha*degree_df["degree_pow"])/sum_dfs
            # list_us = torch.tensor(u).to(self.device).repeat(len(non_local_successors))
            # non_local_edge_ids = self.dgl_g.edge_ids(list_us, torch.tensor(non_local_successors).to(self.device))
            non_local_pr = torch.tensor(list(degree_df.loc[non_local_successors,"degree_pow"]), device=self.device)
            # torch_prs[non_local_edge_ids] = non_local_pr

            list_us = torch.tensor(u).to(self.device).repeat(len(all_successors))
            list_vs = torch.cat((torch.tensor(local_successors),torch.tensor(non_local_successors)))
            list_vs = list_vs.to(torch.int64)
            list_prs = torch.cat((local_pr,non_local_pr))
            
            edge_ids = self.dgl_g.edge_ids(list_us,list_vs.to(self.device))
            torch_prs[edge_ids] = list_prs
        self.dgl_g.edata['p'] = torch_prs
    
            

    def _generate_walks(self) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """
        # Split num_walks for each worker
        nodes = torch.tensor(list(self.graph.nodes()))
        nodes = nodes.repeat(self.num_walks).to(self.device)
        walk_results, _ = dgl.sampling.random_walk(self.dgl_g, nodes=nodes, length=self.walk_len, prob="p")
        print("Walk results shape: ", walk_results.size())
        walks = np.array(walk_results.tolist()).astype(str).tolist()
        return walks

