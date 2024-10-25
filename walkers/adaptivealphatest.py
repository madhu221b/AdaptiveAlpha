from collections import Counter
import networkx as nx
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import copy

try:
  from .walker import Walker
except Exception as error:
    from walker import Walker

class AdaptiveAlphaTest(Walker):
    def __init__(self, graph, beta=0, workers=1, dimensions=64, walk_len=10, num_walks=200):
        print(" Test Adaptive Alpha Non Local In Degree Walker with constant beta: {} And Local Random Walker ".format(beta))
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)

        self.number_of_nodes = self.graph.number_of_nodes()
        self.node_attrs = nx.get_node_attributes(graph, "group")
        self.groups = set(self.node_attrs.values())
 
        # Populate nodes by group
        self._get_group_to_node_dict()
        
        self.d_graph = dict()      
        self.walk_types = ["nonlocal","local"]  
        for node in self.graph.nodes():
            self.d_graph[node] = {"alpha":[0,0]}
            for w_type in self.walk_types:
                self.d_graph[node][w_type] = {"pr":list(), "ngh":list()} 
    
        degree = dict(self.graph.in_degree()) # note now it is indegree
        self.indegree_df = pd.DataFrame.from_dict(degree, orient='index', columns=['degree'])
        degree_pow = dict({node: (np.round(degree**beta,5) if degree != 0 else 0) for node, degree in degree.items()})
        self.degree_pow_df = pd.DataFrame.from_dict(degree_pow, orient='index', columns=['degree_pow'])

    
        print("!!!!  Computing Probability Matrix")
        self._precompute_probabilities()
        print("!!!!  Generate Walks")
        self._generate_walks(self.graph, self.d_graph)
    
    def _get_group_to_node_dict(self):
        self.group_to_node_dict = dict()
        for node, node_id in self.node_attrs.items():
            if node_id not in self.group_to_node_dict:
                self.group_to_node_dict[node_id] = list()
            self.group_to_node_dict[node_id].append(node)

    def _get_non_local_successors(self, node, successors):
        non_local_jump_nodes = list()
        for successor in successors:
            next_succ = self.graph.successors(successor)
            # not already connected to node or is an exisiting successor and is so same identity
            next_succ = [_ for _ in next_succ if _ != node and _ not in successors and self.node_attrs[_]==self.node_attrs[node]]
            non_local_jump_nodes.extend(next_succ)
        
        if len(non_local_jump_nodes) != 0:
            all_nodes = non_local_jump_nodes
        else: 
            all_nodes = self.group_to_node_dict[self.node_attrs[node]]
            all_nodes = list(set(all_nodes) - set(set(successors) | set([node])))
    
        return all_nodes


    def _precompute_probabilities(self):
        for i in self.graph.nodes():
            local_neighbors = list(self.graph.successors(i))
            non_local_neighbors = self._get_non_local_successors(i, local_neighbors)

            unnormalized_prs_local = self.degree_pow_df.loc[local_neighbors, "degree_pow"]
            unnormalized_prs_nonlocal = self.degree_pow_df.loc[non_local_neighbors, "degree_pow"]
            
            # Populating local neighbors and transition prs
            if len(local_neighbors) != 0:
                _sum = 0.0
                for degree, ngh in zip(unnormalized_prs_local,local_neighbors):
                    w = self.graph[i][ngh].get(self.weight_key, 1)
                    num_ = w
                    _sum += num_
                    self.d_graph[i]["local"]["pr"].append(num_)
                    self.d_graph[i]["local"]["ngh"].append(ngh)
                
                self.d_graph[i]["local"]["pr"] = np.array(self.d_graph[i]["local"]["pr"])/_sum
     
            # Populating non-local neighbors and transition prs
            if len(non_local_neighbors) != 0:
                _sum = unnormalized_prs_nonlocal.sum()
                if _sum == 0: 
                    unnormalized_prs_nonlocal = unnormalized_prs_nonlocal + 1e-6
                    _sum = unnormalized_prs_nonlocal.sum()

                prs = unnormalized_prs_nonlocal/_sum
                self.d_graph[i]["nonlocal"]["pr"] = list(prs)
                self.d_graph[i]["nonlocal"]["ngh"] = non_local_neighbors
            
            # Populating walk pr
            het_ngs = len([ _ for _ in local_neighbors if self.node_attrs[_] != self.node_attrs[i]])
            if len(local_neighbors) == 0: 
                self.d_graph[i]["alpha"] = [0.5,0.5]
            else:
               alpha = het_ngs/len(local_neighbors)
               self.d_graph[i]["alpha"] = [alpha,1-alpha]
            # print("i identity:{}, alpha:{}".format(self.node_attrs[i],self.d_graph[i]["alpha"]))

    def _generate_walks(self, graph, d_graph) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)
 
        parallel_generate_walks = self.local_generate_walk

        walk_results = Parallel(n_jobs=self.workers)(
            delayed(parallel_generate_walks)(graph,d_graph,idx, len(num_walks))
                                        for idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)
        self.walks = walks

    def local_generate_walk(self, graph, d_graph, cpu_num, num_walks):
        walks = list()
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))
     
        for n_walk in range(num_walks):
            # random_group = np.random.choice(possible_walks, p=walks_pr, size=1)[0]
            pbar.update(1)

            shuffled_nodes = list(graph.nodes())
            random.shuffle(shuffled_nodes)
         
            # Start a random walk from every node
            for source in shuffled_nodes:
  
                walk = [source]
                walks_pr = d_graph[source]["alpha"]
                random_group = np.random.choice(self.walk_types, p=walks_pr, size=1)[0]
                
                while len(walk) < self.walk_len:
                       last_node = walk[-1]
                       walk_options = list(d_graph[last_node][random_group]["ngh"])
                       probabilities = d_graph[last_node][random_group]["pr"]
                       if len(walk_options) == 0: break
                       next_node = np.random.choice(walk_options, size=1, p=probabilities)[0]
                       walk.append(next_node)

                walk = list(map(str, walk))  # Convert all to strings
                walks.append(walk)

    
        pbar.close()
        return walks