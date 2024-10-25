from tqdm import tqdm
import numpy as np
import random
from joblib import Parallel, delayed
import gensim

class Walker(object):
    def __init__(self, graph,dimensions=64,num_walks=200,walk_len=10,workers=1):
        """Creating a graph."""
        self.weight_key = "weight"
        self.graph = graph
        self.walks = None
        self.dimensions = dimensions
        self.num_walks = num_walks
        self.walk_len = walk_len
        self.workers = workers

    def _generate_walks(self, graph, d_graph, type="local") -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]
       
        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)
        if type == "local":
            parallel_generate_walks = self.local_generate_walk

        
        walk_results = Parallel(n_jobs=self.workers)(
            delayed(parallel_generate_walks)(graph, d_graph, idx, len(num_walks))
                                        for idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)
        self.walks = walks

    def local_generate_walk(self, graph, d_graph, cpu_num, num_walks):
        walks = list()
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

        for n_walk in range(num_walks):
  
            pbar.update(1)

            shuffled_nodes = list(graph.nodes())
            random.shuffle(shuffled_nodes)

            # Start a random walk from every node
            for source in shuffled_nodes:
  
                walk = [source]
                while len(walk) < self.walk_len:
                       last_node = walk[-1]
                       walk_options = list(d_graph[last_node]["ngh"])
                       probabilities = d_graph[last_node]["pr"]
                       if len(probabilities) == 0: break  # skip nodes with no ngs
                       next_node = np.random.choice(walk_options, size=1, p=probabilities)[0]
                       walk.append(next_node)

                walk = list(map(str, walk))  # Convert all to strings
                walks.append(walk)

    
        pbar.close()
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
