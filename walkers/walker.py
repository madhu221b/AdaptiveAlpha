from tqdm import tqdm
import numpy as np
import random
from joblib import Parallel, delayed
import gensim
import torch

# from w2v_cbow.train import generate_embeddings_w2v
# from w2v.train import generate_embeddings_w2v

class Walker(object):
    def __init__(self, graph, dimensions=128,  walk_len=20, walks_per_node=10, workers=1):
        """Creating a graph."""
        self.graph = graph
        self.walks = None
        self.dimensions = dimensions
        self.walk_len = walk_len
        self.walks_per_node = walks_per_node
        self.workers = workers
        self.device =  'cuda' if torch.cuda.is_available() else 'cpu'
   

    def fit(self, **skip_gram_params) -> gensim.models.Word2Vec:
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameteres for gensim.models.Word2Vec - do not supply 'size' it is taken from the FairWalk 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        # if 'workers' not in skip_gram_params:
        #     skip_gram_params['workers'] = self.workers

        # if 'size' not in skip_gram_params:
        #     # skip_gram_params['size'] = self.dimensions
        #     skip_gram_params['vector_size'] = self.dimensions

        skip_gram_params["workers"] = 8
        skip_gram_params["epochs"] = 1
        skip_gram_params["min_count"] = 0
        skip_gram_params["sg"] = 1
        skip_gram_params['vector_size'] = self.dimensions
        skip_gram_params['window'] = 10
     
        return gensim.models.Word2Vec(self.walks, **skip_gram_params)

    # def fit(self):
    #     """
    #     Creates the embeddings using pytorch functionalities.

    #     """
    #     config_dict = {
    #       "embedding_dim": self.dimensions, 
    #       "walk_length":   self.walk_len, 
    #       "context_size":  5,
    #       "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    #       "walks_per_node": self.num_walks, 
    #       "num_negative_samples": 1 # 1 neg sample for every pos sample 

    #       }

    #     # dgl adds "-1" if walks cannot be done , filter those rows
    #     condition = self.walks > 0
    #     row_cond = condition.all(1)
    #     self.walks = self.walks[row_cond, :] 
    #     embeddings = generate_embeddings_w2v(self.walks, self.number_of_nodes, config_dict)
    
    #     return embeddings

    # def fit(self):
    #     """
    #     Creates the embeddings using pytorch functionalities.

    #     """
    #     config_dict = {
    #       "embedding_dim": self.dimensions, 
    #       "walk_length":   self.walk_len, 
    #       "context_size":  5,
    #       "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    #       "walks_per_node": self.num_walks, 
    #       "num_negative_samples": 3 # 1 neg sample for every pos sample 

    #       }

    #     # dgl adds "-1" if walks cannot be done , filter those rows
    #     condition = self.walks > 0
    #     row_cond = condition.all(1)
    #     self.walks = self.walks[row_cond, :] 
    #     embeddings = generate_embeddings_w2v(self.walks, self.number_of_nodes, config_dict)
    
    #     return embeddings

