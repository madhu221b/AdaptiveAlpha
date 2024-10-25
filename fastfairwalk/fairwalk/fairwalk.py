import os
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import gensim
from joblib import Parallel, delayed
from tqdm import tqdm

from .parallel import parallel_generate_walks

import torch
import dgl
from dgl.sampling import random_walk
"""
This version only works for p=1, q=1 (Deepwalk-like walks)

"""
class FastFairWalk:
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    GROUP_KEY = 'group'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self, graph: nx.Graph, dimensions: int = 128, walk_length: int = 80, num_walks: int = 10, p: float = 1,
                 q: float = 1, weight_key: str = 'weight', workers: int = 1, sampling_strategy: dict = None,
                 quiet: bool = False, temp_folder: str = None):
        """
        Initiates the FairWalk object, precomputes walking probabilities and generates the walks.

        :param graph: Input graph
        :param dimensions: Embedding dimensions (default: 128)
        :param walk_length: Number of nodes in each walk (default: 80)
        :param num_walks: Number of walks per node (default: 10)
        :param p: Return hyper parameter (default: 1)
        :param q: Inout parameter (default: 1)
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :param workers: Number of workers for parallel execution (default: 1)
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        :param temp_folder: Path to folder with enough space to hold the memory map of self.d_graph (for big graphs); to be passed joblib.Parallel.temp_folder
        """

        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.quiet = quiet
        self.d_graph = defaultdict(dict)

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.temp_folder, self.require = None, None
        if temp_folder:
            if not os.path.isdir(temp_folder):
                raise NotADirectoryError("temp_folder does not exist or is not a directory. ({})".format(temp_folder))

            self.temp_folder = temp_folder
            self.require = "sharedmem"
        
        self.device =  'cuda' if torch.cuda.is_available() else 'cpu'


        self.node_attr = nx.get_node_attributes(self.graph,"group")
        self.group_df = pd.DataFrame.from_dict(self.node_attr, orient='index', columns=['group'])

        print("Computing node embeddings on {}".format(self.device))
        print("!! Obtain DGL Graph from Networkx")
        self._precompute_graph() # convert to dgl graph
        print("!! Precomputing Probablities - fw (fair group selection + n2v (p=1,q=1))")
        self._precompute_probabilities() # populate d_graph
        print("!! Generating walks")
        self.walks = self._generate_walks()

    
    def _precompute_graph(self):
        self.dgl_g = dgl.from_networkx(self.graph, node_attrs=['group'])
        self.dgl_g.edata['p'] = torch.zeros(self.dgl_g.num_edges(), dtype=torch.float32)
        self.dgl_g = self.dgl_g.to(self.device)


    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node. 
        Store in dgl's graph edata attributes.
        >> Need to store unnormalized prs
        """
        us, vs, ps = list(), list(), list()
        torch_prs = torch.zeros(self.dgl_g.num_edges(), dtype=torch.float32, device=self.device)
        
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

   
            list_us = torch.tensor(u).to(self.device).repeat(len(successors))
            list_vs = torch.tensor(successors).to(self.device)
            list_prs = torch.tensor(list(join_df.loc[successors, "total_pr"])).to(self.device)
            edge_ids = self.dgl_g.edge_ids(list_us,list_vs)
            torch_prs[edge_ids] = list_prs
        
        print("have to copy e data")
        self.dgl_g.edata['p'] = torch_prs
        print("done copying e data")


    def _generate_walks(self) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        # Split num_walks for each worker
        nodes = torch.tensor(list(self.graph.nodes()))
        nodes = nodes.repeat(self.num_walks).to(self.device)
        print("Doing Random Walks")
        walk_results, _ = dgl.sampling.random_walk(self.dgl_g, nodes=nodes, length=self.walk_length, prob="p")
        print("Walk results shape: ", walk_results.size())
        walks = np.array(walk_results.tolist()).astype(str).tolist()
        return walks

    def fit(self, **skip_gram_params) -> gensim.models.Word2Vec:
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameteres for gensim.models.Word2Vec - do not supply 'size' it is taken from the FairWalk 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        if 'size' not in skip_gram_params:
            # skip_gram_params['size'] = self.dimensions
            skip_gram_params['vector_size'] = self.dimensions
    
        return gensim.models.Word2Vec(self.walks, **skip_gram_params)
