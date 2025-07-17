from collections import Counter, defaultdict
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

class FastFairWalk(Walker):
    def __init__(self, graph, workers=1, dimensions=128, walk_len=20, walks_per_node=10, p=1, q=1):
        print(f"Fast Fairwalk Walker - with assumed p={p}, q={q}")
        super().__init__(graph, workers=workers, dimensions=dimensions, walk_len=walk_len, walks_per_node=walks_per_node)
        self.quiet = False
        self.is_optimise = False # For huge graphs, relying less on networkx
        self.number_of_nodes = self.graph.number_of_nodes()
        
        self.device = "cpu"
        self.d_graph = defaultdict(dict)
        self.FIRST_TRAVEL_KEY = 'first_travel_key'
        self.PROBABILITIES_KEY = 'probabilities'
        self.NEIGHBORS_KEY = 'neighbors'
        self.GROUP_KEY = 'group'
        self.p, self.q = p, q
        self.sampling_strategy = {}
      

    
        print("Computing node embeddings on {}".format(self.device))
        print("!! Precomputing Probablities")
        self._precompute_probabilities() # populate d_graph
        print("!!!!  Generate Walks")
        self.walks = self._generate_walks()
    
    

    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """
        p = self.p
        q = self.q


        d_graph = self.d_graph

        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

        node2groups = nx.get_node_attributes(self.graph, self.GROUP_KEY)
        groups = np.unique(list(node2groups.values()))

        # Init probabilities dict
        for node in self.graph.nodes():
            for group in groups:
                if self.PROBABILITIES_KEY not in d_graph[node]:
                    d_graph[node][self.PROBABILITIES_KEY] = dict()
                if group not in d_graph[node][self.PROBABILITIES_KEY]:
                    d_graph[node][self.PROBABILITIES_KEY][group] = dict()

        for source in nodes_generator:
            for current_node in self.graph.neighbors(source):

                unnormalized_weights = list()
                d_neighbors = list()
                neighbor_groups = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):
                    weight = 1 # unweighted graph
                    if destination == source:  # Backwards probability
                        ss_weight = weight * 1 / p
                    elif destination in self.graph[source]:  # If the neighbor is connected to the source
                        ss_weight = weight
                    else:
                        ss_weight = weight * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    d_neighbors.append(destination)
                    neighbor_groups.append(self.graph.nodes[destination][self.GROUP_KEY])

                unnormalized_weights = np.array(unnormalized_weights)
                d_neighbors = np.array(d_neighbors)
                neighbor_groups = np.array(neighbor_groups)

                for group in groups:
                    cur_unnormalized_weights = unnormalized_weights[neighbor_groups == group]
                    cur_d_neighbors = d_neighbors[neighbor_groups == group]

                    # Normalize
                    d_graph[current_node][self.PROBABILITIES_KEY][group][
                        source] = cur_unnormalized_weights / cur_unnormalized_weights.sum()

                    # Save neighbors
                    d_graph[current_node].setdefault(self.NEIGHBORS_KEY, {})[group] = list(cur_d_neighbors)

            # Calculate first_travel weights for source
            first_travel_weights = []
            first_travel_neighbor_groups = []
            for destination in self.graph.neighbors(source):
                first_travel_weights.append(1) # unweighted graph
                first_travel_neighbor_groups.append(self.graph.nodes[destination][self.GROUP_KEY])

            first_travel_weights = np.array(first_travel_weights)
            first_travel_neighbor_groups = np.array(first_travel_neighbor_groups)
            d_graph[source][self.FIRST_TRAVEL_KEY] = {}
            for group in groups:
                cur_first_travel_weights = first_travel_weights[first_travel_neighbor_groups == group]
                d_graph[source][self.FIRST_TRAVEL_KEY][group] = cur_first_travel_weights / cur_first_travel_weights.sum()

    def _generate_walks(self) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.walks_per_node), self.workers)

        walk_results = Parallel(n_jobs=self.workers)(
            delayed(parallel_generate_walks)(self.d_graph,
                                             self.walk_len,
                                             len(num_walks),
                                             idx,
                                             self.sampling_strategy,
                                             self.NEIGHBORS_KEY,
                                             self.PROBABILITIES_KEY,
                                             self.FIRST_TRAVEL_KEY,
                                             self.quiet) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)
        walks = [[str(node) for node in walk] for walk in walks]
        print(f"Walk results shape:({len(walks)},{len(walks[0])}) ", )

        return walks



def parallel_generate_walks(d_graph: dict, global_walk_length: int, num_walks: int, cpu_num: int,
                            sampling_strategy: dict = None, 
                            neighbors_key: str = None, probabilities_key: str = None,
                            first_travel_key: str = None, quiet: bool = False) -> list:
    """
    Generates the random walks which will be used as the skip-gram input.

    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:



            # Start walk
            walk = [source]

            # Calculate walk length
            walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                # Skip dead end nodes
                if not walk_options:
                    break

                group2neighbors = d_graph[walk[-1]][neighbors_key]
                all_possible_groups = [group for group in group2neighbors if len(group2neighbors[group]) > 0]
                if not all_possible_groups: break
                random_group = np.random.choice(all_possible_groups, size=1)[0]
                walk_options = walk_options[random_group]

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key][random_group]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][random_group][walk[-2]]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]

                walk.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks