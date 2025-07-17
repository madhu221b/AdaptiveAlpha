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

class FastAdaptiveAlphaTest(Walker):
    def __init__(self, graph, beta=0, workers=1, dimensions=64, walk_len=10, num_walks=200):
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)
        print(" FAST Test Adaptive Alpha Non Local In Degree Walker with constant beta: {} And Local Random Walker ".format(beta))
        self.quiet = False
        self.is_optimise = False # For huge graphs, relying less on networkx
        self.number_of_nodes = self.graph.number_of_nodes()
        
        # self.device =  'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = "cpu"
        self.edge_dict = dict()
      
        self.node_attrs = nx.get_node_attributes(self.graph, "group")
        self.group_df = pd.DataFrame.from_dict(self.node_attrs, orient='index', columns=['group'])
        self.groups = set(self.node_attrs.values())

        degree = dict(self.graph.in_degree()) # note now it is indegree
        self.indegree_df = pd.DataFrame.from_dict(degree, orient='index', columns=['degree'])
        degree_pow = dict({node: (np.round(degree**beta,5) if degree != 0 else 0) for node, degree in degree.items()})
        self.degree_pow_df = pd.DataFrame.from_dict(degree_pow, orient='index', columns=['degree_pow'])
      
 
        # Populate nodes by group
        self._get_group_to_node_dict()
        self.avg_out_degree_by_group = self._get_avg_outdegree()
        print("mx degree by group: ", self.avg_out_degree_by_group)

    

    
        print("Computing node embeddings on {}".format(self.device))
        print("!! Obtain DGL Graph from Networkx")
        self._precompute_graph() # convert to dgl graph
        print("!! Precomputing Probablities -  Adaptive Alpha - NonLocal ID + Local RW (p=1,q=1)")
        self._precompute_probabilities() # populate d_graph
        print("!!!!  Generate Walks")
        self.walks = self._generate_walks()
    

    def _get_avg_outdegree(self):
        out_degree = dict()
        degree = dict(self.graph.out_degree()) # note now it is indegree
        outdegree_df = pd.DataFrame.from_dict(degree, orient='index', columns=['outdegree'])
        for group in self.groups:
            nodes_by_group = list(self.group_df.loc[self.group_df["group"] == group,:].index)
            outdegree_by_degree = outdegree_df.loc[nodes_by_group, "outdegree"]
            avg_outd = outdegree_by_degree.max() # using max
            out_degree[group] = avg_outd


        return out_degree

    def _get_group_to_node_dict(self):

        self.group_to_node_dict = dict()
        # populate by 
        for group in self.groups:
            nodes_by_group = list(self.group_df.loc[self.group_df["group"] == group,:].index)
            prs = self.degree_pow_df.loc[nodes_by_group, "degree_pow"]
            prs += 1e-6
            sum_prs = prs.sum()
            prs = prs/sum_prs
            self.group_to_node_dict[group] = dict()
            self.group_to_node_dict[group]["samples"] = nodes_by_group
            self.group_to_node_dict[group]["pr"] = prs



    def _get_non_local_successors(self, node, successors):
        non_local_jump_nodes = list()
        id_u = self.node_attrs[node]

        for successor in successors:
            next_succ = self.graph.successors(successor)
            # not already connected to node or is an exisiting successor and is so same identity
            next_succ = [_ for _ in next_succ if _ != node and _ not in successors and self.node_attrs[_]==self.node_attrs[node]]
            non_local_jump_nodes.extend(next_succ)

        if len(non_local_jump_nodes) != 0:
            all_nodes = non_local_jump_nodes
        else: 
            all_nodes = self.group_to_node_dict[id_u]["samples"]
            all_pr = self.group_to_node_dict[id_u]["pr"]
            size = min(len(all_nodes), self.avg_out_degree_by_group[id_u])
            all_nodes = np.random.choice(all_nodes, size=size, p=all_pr, replace=False)
            all_nodes = list(set(all_nodes) - set(set(successors) | set([node])))
    
        return all_nodes
    
        
    def _precompute_graph(self):
        # find all non-local edges
        print("find non-local edges in networkx")
        if self.is_optimise:  self.dgl_g = dgl.from_networkx(self.graph, node_attrs=['group'])
        edge_list = list()
        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing non-local edges')

        
        tensor_u, tensor_v = list(), list()
        for u in nodes_generator:
            successors = list(self.graph.successors(u))
            non_local_successors = self._get_non_local_successors(u, successors)
            if len(successors) == 0:
                print("for node {}, number of successors: {}, nonlocal successors: {}".format(u,len(successors),len(non_local_successors)))
            self.edge_dict[u] = {"local":successors, "nonlocal":non_local_successors}
            
            # Populating non-local edges in data structures
            if self.is_optimise is False:
                edge_list.extend([(u,v) for v in non_local_successors])
            else:
                tensor_u.extend([u]*len(non_local_successors))
                tensor_v.extend(non_local_successors)
        

        if self.is_optimise is False: # add edges in networkx
             print("Adding edges in networkx graph")
             self.clone_graph = self.graph.copy()
             self.clone_graph.add_edges_from(edge_list)  # this takes time in networkx if graph is huge
             self.dgl_g = dgl.from_networkx(self.clone_graph) 
             self.dgl_g = self.dgl_g.to(self.device)
        else: # add edges in dgl graph
            print("Adding edges in dgl graph, initially edges", self.dgl_g.num_edges())
            self.dgl_g.add_edges(tensor_u, tensor_v)
            print("After adding non-local edges", self.dgl_g.num_edges())

        
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
            local_successors, non_local_successors = self.edge_dict[u]["local"], self.edge_dict[u]["nonlocal"]
            # assert len(all_successors) == len(local_successors) + len(non_local_successors)
            len_local, len_nonlocal = len(local_successors), len(non_local_successors)
            len_all = len_local + len_nonlocal
            
            # calculate alpha
            id_u = self.node_attrs[u]
            len_v = len([_ for _ in local_successors if self.node_attrs[_] != id_u])
            if len_local == 0: alpha = 1.0
            else: alpha = len_v/len_local
            
            # assign local-weight
            # if len_local != 0: local_pr = 1/len_local
            # else: local_pr = 0
            # one_minus_alpha  = (1-alpha)
            # pr = torch.tensor([one_minus_alpha*local_pr], device=device)
            # local_pr = pr.repeat(len_local)

            ## assign local weight
            one_minus_alpha  = (1-alpha)
            if len_local != 0: 
                local_degree_df = self.degree_pow_df.loc[local_successors, :]
                sum_dfs_l = local_degree_df['degree_pow'].sum()
                # normalize the column and multiply by alpha
                local_degree_df["degree_pow"] = (one_minus_alpha*local_degree_df["degree_pow"])/sum_dfs_l
                local_pr = torch.tensor(list(local_degree_df['degree_pow']), device=device)
            else: 
                local_pr = torch.tensor([], device=device)
  
        

            # assign non-local weight          
            degree_df = self.degree_pow_df.loc[non_local_successors, :]
            sum_dfs = degree_df['degree_pow'].sum()
            # normalize the column and multiply by alpha
            degree_df["degree_pow"] = (alpha*degree_df["degree_pow"])/sum_dfs
            non_local_pr = torch.tensor(list(degree_df['degree_pow']), device=device)


            list_us = torch.tensor(u).to(device).repeat(len_all)
            list_vs = torch.cat((torch.tensor(local_successors),torch.tensor(list(degree_df.index))))
            list_vs = list_vs.to(torch.int64).to(device)
            # print("local pr size: {}, non local pr size: {}".format(local_pr.size(), non_local_pr.size()))
            list_prs = torch.cat((local_pr,non_local_pr))
            
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

