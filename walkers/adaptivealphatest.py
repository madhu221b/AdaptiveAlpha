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
        print(" Test Adaptive Alpha In Degree Walker with constant beta: ", beta)
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)

        self.number_of_nodes = self.graph.number_of_nodes()
        self.node_attrs = nx.get_node_attributes(graph, "group")
        self.groups = set(self.node_attrs.values())
 
        self.group_to_node_dict = self._get_group_to_node_dict()


        self.d_graph = dict()      
        walk_types = ["local","nonlocal"]  
        for node in self.graph.nodes():
            self.d_graph[node] = dict()
            for w_type in walk_types:
                self.d_graph[node][w_type] = {"pr":list(), "ngh":list()}
    
        degree = dict(self.graph.in_degree()) # note now it is indegree
        self.indegree_df = pd.DataFrame.from_dict(degree, orient='index', columns=['degree'])
        degree_pow = dict({node: (np.round(degree**beta,5) if degree != 0 else 0) for node, degree in degree.items()})
        self.degree_pow_df = pd.DataFrame.from_dict(degree_pow, orient='index', columns=['degree_pow'])


        print("!!!! Computing non-local jump probability")
        self.walk_alpha_pr = dict()
        self._precompute_alpha()
        

        print("!!!!  Computing Probability Matrix")
        self._precompute_probabilities()
        print("!!!!  Generate Walks")
        self._generate_walks(self.graph, self.d_graph, self.walk_alpha_pr)
      
    def avg_indegree_due_to_itself(self, grp):
        g = self.graph
        itr = [node for node, _ in self.node_attrs.items() if _ == grp]
        total_len = len(itr)
        sum_ = 0
        total_sum_ = 0
        for i in itr:      
            neighbors = list(g.predecessors(i))
            diff_nghs = len([ngh for ngh in neighbors if self.node_attrs[ngh] == grp])
            sum_ += diff_nghs
            total_sum_ += len(neighbors)
        if total_sum_ == 0: return sum_
        avg_indg = sum_/total_sum_
        return avg_indg

    def avg_outdegree_due_to_itself(self, grp):
        g = self.graph
        itr = [node for node, _ in self.node_attrs.items() if _ == grp]
        total_len = len(itr)
        sum_ = 0
        total_sum_ = 0
        for i in itr:      
            neighbors = list(g.successors(i))
            diff_nghs = len([ngh for ngh in neighbors if self.node_attrs[ngh] == grp])
            sum_ += diff_nghs
            total_sum_ += len(neighbors)
        if total_sum_ == 0: return sum_
        avg_outdg = sum_/total_sum_
        return avg_outdg

    def _precompute_alpha(self):
        epsilon = 1e-3
        h_o_g = dict()
        for u_group in self.groups:
            term1 =  self.avg_indegree_due_to_itself(u_group) + epsilon
            term2 =  self.avg_outdegree_due_to_itself(u_group) + epsilon
            product = 1/(term1 * term2)
            h_o_g[u_group] = product
            

        sum_hog = sum((list(h_o_g.values())))
        h_o_g = {k:v/sum_hog for k,v in h_o_g.items()}

        # for node in self.graph.nodes():
        #     self.walk_alpha_pr[node] = h_o_g[self.node_attrs[node]]
            
        self.walk_alpha_pr = h_o_g
        print("Alpha prs: ", self.walk_alpha_pr)
    
 
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

    def _get_group_to_node_dict(self):
        group2node = dict()
        for node, node_id in self.node_attrs.items():
            if node_id not in group2node: group2node[node_id] = list()
            group2node[node_id].append(node)  
        return group2node

    def _precompute_probabilities(self):
        for i in self.graph.nodes():
            local_neighbors = list(self.graph.successors(i))
            non_local_neighbors = self._get_non_local_successors(i, local_neighbors)

            unnormalized_prs_local = self.degree_pow_df.loc[local_neighbors, "degree_pow"]
            unnormalized_prs_nonlocal = self.degree_pow_df.loc[non_local_neighbors, "degree_pow"]

            if len(local_neighbors) != 0:
                _sum = 0.0
                for degree, ngh in zip(unnormalized_prs_local,local_neighbors):
                    w = self.graph[i][ngh].get(self.weight_key, 1)
                    num_ = w*degree
                    _sum += num_
                    self.d_graph[i]["local"]["pr"].append(num_)
                    self.d_graph[i]["local"]["ngh"].append(ngh)
                
                self.d_graph[i]["local"]["pr"] = np.array(self.d_graph[i]["local"]["pr"])/_sum
     

            if len(non_local_neighbors) != 0:
                _sum = unnormalized_prs_nonlocal.sum()
                if _sum == 0: 
                    unnormalized_prs_nonlocal = unnormalized_prs_nonlocal + 1e-6
                    _sum = unnormalized_prs_nonlocal.sum()

                prs = unnormalized_prs_nonlocal/_sum
                self.d_graph[i]["nonlocal"]["pr"] = list(prs)
                self.d_graph[i]["nonlocal"]["ngh"] = non_local_neighbors



    def _generate_walks(self, graph, d_graph, walk_alpha_pr) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        self.num_walks = 100 # changing here
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)
 
        parallel_generate_walks = self.local_generate_walk

        walk_results = Parallel(n_jobs=self.workers)(
            delayed(parallel_generate_walks)(graph, d_graph,walk_alpha_pr, idx, len(num_walks))
                                        for idx, num_walks
            in enumerate(num_walks_lists, 1))
        walks = flatten(walk_results)
        
        parallel_generate_walks = self.non_local_generate_walk
        
        for group in self.groups:
            
            nodes_group = self.group_to_node_dict[group]
            num_walks = int(self.num_walks*self.walk_alpha_pr[group])
            print("Generate non local walks for group : {}, walks: {} ".format(group,num_walks))
            num_walks_lists = np.array_split(range(num_walks), self.workers)
            walk_results = Parallel(n_jobs=self.workers)(
                           delayed(parallel_generate_walks)(graph,d_graph,nodes_group,idx,len(num_walks))
                                        for idx, num_walks
            in enumerate(num_walks_lists, 1))
            non_local_walks = flatten(walk_results)
            walks.extend(non_local_walks)
            
        self.walks = walks

    def local_generate_walk(self, graph, d_graph, walk_alpha_pr, cpu_num, num_walks):
        walks = list()
        pbar = tqdm(total=num_walks, desc='Generating LOCAL walks (CPU: {}), num_walks: {}'.format(cpu_num,num_walks))

        for n_walk in range(num_walks):
  
            pbar.update(1)

            shuffled_nodes = list(graph.nodes())
            random.shuffle(shuffled_nodes)

            # Start a random walk from every node
            for source in shuffled_nodes:
  
                walk = [source]
                while len(walk) < self.walk_len:
                       last_node = walk[-1]
                       walk_options = list(d_graph[last_node]["local"]["ngh"])
                       probabilities = d_graph[last_node]["local"]["pr"]
                       if len(probabilities) == 0: break  # skip nodes with no ngs
                       next_node = np.random.choice(walk_options, size=1, p=probabilities)[0]
                       walk.append(next_node)

                walk = list(map(str, walk))  # Convert all to strings
                walks.append(walk)
                
        pbar.close()
        return walks

    def non_local_generate_walk(self, graph, d_graph, nodes_group, cpu_num, num_walks):
        walks = list()
        pbar = tqdm(total=num_walks, desc='Generating NON-LOCAL walks (CPU: {}), num_walks: {}'.format(cpu_num,num_walks))

        for n_walk in range(num_walks):
  
            pbar.update(1)

            shuffled_nodes = nodes_group
            random.shuffle(shuffled_nodes)

            # Start a random walk from every node
            for source in shuffled_nodes:
  
                walk = [source]
                while len(walk) < self.walk_len:
                       last_node = walk[-1]
                       walk_options = list(d_graph[last_node]["nonlocal"]["ngh"])
                       probabilities = d_graph[last_node]["nonlocal"]["pr"]
                       if len(probabilities) == 0: break  # skip nodes with no ngs
                       next_node = np.random.choice(walk_options, size=1, p=probabilities)[0]
                       walk.append(next_node)

                walk = list(map(str, walk))  # Convert all to strings
                walks.append(walk)
                
        pbar.close()
        return walks