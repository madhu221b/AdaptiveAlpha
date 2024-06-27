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

class AdaptiveAlpha(Walker):
    def __init__(self,graph,beta=0,workers=1,dimensions=64,walk_len=10,num_walks=200):
        print(" [V2] Adaptive Alpha In Degree Walker with constant beta: ", beta)
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)

        self.number_of_nodes = self.graph.number_of_nodes()
        self.node_attrs = nx.get_node_attributes(graph, "group")
        self.groups = set(self.node_attrs.values())
 
        # Populate nodes by group
        self._get_group_to_node_dict()

        # Transition Prs matrix
        # self.pi = np.zeros((self.number_of_nodes, self.number_of_nodes))
        walk_types =  ["local","nonlocal"]
        self.d_graph = dict()
        
        
        for node in self.graph.nodes():
            self.d_graph[node] = dict()
            for w_type in walk_types:
                self.d_graph[node][w_type] = {"pr":list(), "ngh":list()}
    
        degree = dict(self.graph.in_degree()) # note now it is indegree
        self.indegree_df = pd.DataFrame.from_dict(degree, orient='index', columns=['degree'])
        degree_pow = dict({node: (np.round(degree**beta,5) if degree != 0 else 0) for node, degree in degree.items()})
        self.degree_pow_df = pd.DataFrame.from_dict(degree_pow, orient='index', columns=['degree_pow'])

        # computing local group prs
        print("!!!! Computing Local Group Selection probability")
        self.group2pr = self._get_group_selection_prs()
        # compute probabilities

        print("!!!! Computing non-local jump probability")
        self.walk_alpha_pr = dict()
        self._precompute_alpha()
        

        print("!!!!  Computing Probability Matrix")
        self._precompute_probabilities()
        print("!!!!  Generate Walks")
        self._generate_walks(self.graph, self.d_graph, self.walk_alpha_pr)
    

    def _get_group_to_node_dict(self):
        self.group2node = dict()
        for node, node_id in self.node_attrs.items():
            if node_id not in self.group2node:
                self.group2node[node_id] = list()
            self.group2node[node_id].append(node)
    
    def _get_group_selection_prs(self):
        group_dict = dict()
        local_walk_dict = dict()

        for group in self.groups:
            o_g_g = self.avg_outdegree_due_to_itself(group)
            o_g_gbar = self.avg_outdegree_to_grp_dict(group)
            o_g_gbar.update({group:o_g_g})
           
            if group not in group_dict: group_dict[group] = {"groups":[], "pr":[]}
            for k,v in o_g_gbar.items():
                homo = self.avg_outdegree_due_to_itself(k)*self.avg_indegree_due_to_itself(k)*(1-v)
                group_dict[group]["groups"].append(k)
                group_dict[group]["pr"].append(homo)

 
        self.local_walk_dict = copy.deepcopy(group_dict)
        

        for group in group_dict:
            sum_ = np.sum(np.array(group_dict[group]["pr"]))
            if sum_ != 0:
              group_dict[group]["pr"] = np.array(group_dict[group]["pr"])/sum_
            elif sum_ == 0 and len(group_dict[group]["pr"]) == 1:
              group_dict[group]["pr"] = np.full_like(np.array(group_dict[group]["pr"]), 1)
            else:
                group_dict[group]["pr"] = np.full_like(np.array(group_dict[group]["pr"]), 0.5)

        return group_dict

    def _get_edge_dict(self):
        g = self.graph
        node_attrs = self.node_attrs
        edge_dict = dict()

        for u,v in g.edges():
            label = "{}->{}".format(node_attrs[u],node_attrs[v])
            if label not in edge_dict: edge_dict[label] = 0
            edge_dict[label] += 1

        return edge_dict
   
    def _compute_homophily(self):
        g = self.graph
        edge_dict = self._get_edge_dict()
        groups = self.groups
        homo_dict = dict()
        for uniquegroup in groups:
            denominator = sum([edge_dict["{}->{}".format(uniquegroup,grp)] for grp in groups])
            homo_dict[uniquegroup] = edge_dict["{}->{}".format(uniquegroup,uniquegroup)]/denominator
        return homo_dict


    def avg_indegree_due_to_grp(self, grp):
        g = self.graph
        itr = [node for node, _ in self.node_attrs.items() if _ == grp]
        total_len = len(itr)
        sum_ = 0
        total_sum_ = 0
        for i in itr:      
            neighbors = list(g.predecessors(i))
            diff_nghs = len([ngh for ngh in neighbors if self.node_attrs[ngh] != grp])
            sum_ += diff_nghs
            total_sum_ += len(neighbors)
        if total_sum_ == 0: return sum_
        avg_indg = sum_/total_sum_
        return avg_indg

    def avg_outdegree_to_grp(self, grp):
        g = self.graph
        itr = [node for node, _ in self.node_attrs.items() if _ == grp]
        total_len = len(itr)
        sum_ = 0
        total_sum_ = 0
        for i in itr:      
            neighbors = list(g.successors(i))
            diff_nghs = len([ngh for ngh in neighbors if self.node_attrs[ngh] != grp])
            sum_ += diff_nghs
            total_sum_ += len(neighbors)
        if total_sum_ == 0: return sum_
        avg_outdg = sum_/total_sum_
        return avg_outdg
     
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

    def avg_outdegree_to_grp_dict(self, grp):
        g = self.graph
        itr = [node for node, _ in self.node_attrs.items() if _ == grp]
        total_len = len(itr)
        sum_ = 0
        total_sum_ = 0
        out_dict = dict()

        for i in itr:      
            neighbors = list(g.successors(i))
            diff_nghs = [ngh for ngh in neighbors if self.node_attrs[ngh] != grp]
            
            for diff_ngh in diff_nghs:
                id_ = self.node_attrs[diff_ngh]
                if id_ not in out_dict: out_dict[id_] = 0
                out_dict[id_] += 1 
            
            total_sum_ += len(neighbors)
        
        avg_outdg = {k:v if v==0 else v/total_sum_ for k,v in out_dict.items()}
        return avg_outdg

    def avg_indegree_to_grp_dict(self, grp):
        g = self.graph
        itr = [node for node, _ in self.node_attrs.items() if _ == grp]
        total_len = len(itr)
        sum_ = 0
        total_sum_ = 0
        in_dict = dict()

        for i in itr:      
            neighbors = list(g.predecessors(i))
            diff_nghs = [ngh for ngh in neighbors if self.node_attrs[ngh] != grp]
            
            for diff_ngh in diff_nghs:
                id_ = self.node_attrs[diff_ngh]
                if id_ not in in_dict: in_dict[id_] = 0
                in_dict[id_] += 1 
            
            total_sum_ += len(neighbors)
        
        avg_indg = {k:v if v==0 else v/total_sum_ for k,v in in_dict.items()}
        return avg_indg

    def _precompute_alpha(self):
        uniquegroups = self.group2node.keys()
        group2alpha = dict()
        epsilon = 1e-3
        edge_dict = self._get_edge_dict()


        same_dict = dict()
        for uniquegroup in uniquegroups:
            out_i = self.avg_outdegree_due_to_itself(uniquegroup)
            in_i = self.avg_indegree_due_to_itself(uniquegroup)

            out_g = self.avg_outdegree_to_grp(uniquegroup)
            in_g = self.avg_indegree_due_to_grp(uniquegroup)

            same_dict[uniquegroup] = { "out_i":out_i, "in_i":in_i,
                                       "out_g": out_g, "in_g":in_g }
            

        for uniquegroup in uniquegroups:
            u_dict = self.avg_outdegree_to_grp_dict(uniquegroup)
            v_dict = self.avg_indegree_to_grp_dict(uniquegroup)
        
            print("u_dict: {}, v_dict: {}".format(u_dict,v_dict))
            un_norm_NL = 0
            for k, _ in u_dict.items():
                q_out_gbar, q_in_gbar = u_dict.get(k,0), v_dict.get(k,0)
                term1 = q_in_gbar * same_dict[uniquegroup]["out_i"]
                term2 = q_out_gbar * same_dict[uniquegroup]["in_i"]
                un_norm_NL += (term1+term2)
            
            if len(u_dict):
                un_norm_NL = un_norm_NL/len(u_dict)
         
            # same_dict_grp = {k:v for k,v in same_dict.items() if k != uniquegroup}
            len_ = len(same_dict)
            print("!!", self.local_walk_dict[uniquegroup])
            un_norm_L  = np.sum(self.local_walk_dict[uniquegroup]["pr"])
            # un_norm_L = un_norm_L/len(self.local_walk_dict[uniquegroup]["pr"])
            
            if un_norm_L == 0 and un_norm_NL == 0:
                un_norm_L = un_norm_NL = 0.5

            sum_ = un_norm_L+un_norm_NL
            print("unnormal L : {}, unnormal NL: {}".format(un_norm_L,un_norm_NL))
            if uniquegroup not in group2alpha: group2alpha[uniquegroup] = dict()
            group2alpha[uniquegroup]["local"] = un_norm_L/sum_
            group2alpha[uniquegroup]["nonlocal"] =  un_norm_NL/sum_
            print("Normalised prs: ", group2alpha[uniquegroup])


        print("Group2Alpha: ", group2alpha)
        for i in self.graph.nodes():
            self.walk_alpha_pr[i] = group2alpha[self.node_attrs[i]]
   
    
    def _get_non_local_successors(self, node, successors):
        """
        Sampling with 
        max_val = max(group_size of nghs)
        sample size = max val - group size of group i in ngh
        Design choice - nghs picked at random 

        this wont work.

        Trying another approach - degree wise
        """
        k = 5
       
        non_local_jump_nodes = list()
        for successor in successors:
            next_succ = self.graph.successors(successor)
            # not already connected to node or is an exisiting successor and is so same identity
            next_succ = [_ for _ in next_succ if _ != node and _ not in successors and self.node_attrs[_]==self.node_attrs[node]]
            non_local_jump_nodes.extend(next_succ)
        
        if len(non_local_jump_nodes) != 0:
            all_nodes = non_local_jump_nodes
        else: 
            all_nodes = self.group2node[self.node_attrs[node]]
            all_nodes = list(set(all_nodes) - set(set(successors) | set([node])))
            
        sample_size = min(len(successors),len(all_nodes))
        if sample_size == 0: sample_size = k
        unnormalized_prs = self.degree_pow_df.loc[all_nodes, "degree_pow"]
        unnormalized_prs += 1e-6
        _sum = unnormalized_prs.sum()
        prs = unnormalized_prs/_sum
        non_local_nodes = np.random.choice(all_nodes, size=sample_size, p=prs, replace=False)

        return non_local_nodes
        
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
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)
 
        parallel_generate_walks = self.local_generate_walk

        walk_results = Parallel(n_jobs=self.workers)(
            delayed(parallel_generate_walks)(graph, d_graph,walk_alpha_pr, idx, len(num_walks))
                                        for idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)
        self.walks = walks

    def local_generate_walk(self, graph, d_graph, walk_alpha_pr, cpu_num, num_walks):
        walks = list()
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))
        possible_walks = ["local", "nonlocal"]
        # walks_pr = [1-alpha, alpha] # pr of selecting high degree nodes, low degree nodes

        for n_walk in range(num_walks):
            # random_group = np.random.choice(possible_walks, p=walks_pr, size=1)[0]
            pbar.update(1)

            shuffled_nodes = list(graph.nodes())
            random.shuffle(shuffled_nodes)
         
            # Start a random walk from every node
            for source in shuffled_nodes:
  
                walk = [source]
                # alpha = walk_alpha_pr[source]
                # walks_pr = [1-alpha, alpha]
                walks_pr = [walk_alpha_pr[source][possible_walks[0]], walk_alpha_pr[source][possible_walks[1]]]
                random_group = np.random.choice(possible_walks, p=walks_pr, size=1)[0]
                
                while len(walk) < self.walk_len:
                       last_node = walk[-1]
                       walkgroups = [group for group in possible_walks if len(d_graph[last_node][group]["pr"]) > 0]
                       
                       
                       if random_group not in walkgroups: break

                       walk_options = list(d_graph[last_node][random_group]["ngh"])
                       probabilities = d_graph[last_node][random_group]["pr"]
                       next_node = np.random.choice(walk_options, size=1, p=probabilities)[0]
                       walk.append(next_node)

                walk = list(map(str, walk))  # Convert all to strings
                walks.append(walk)

    
        pbar.close()
        return walks