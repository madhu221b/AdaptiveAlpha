import random
import numpy as np
from tqdm import tqdm


def parallel_generate_walks(d_graph: dict, global_walk_length: int, num_walks: int, cpu_num: int,
                            sampling_strategy: dict = None, num_walks_key: str = None, walk_length_key: str = None,
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

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                # Skip dead end nodes
                if not walk_options:
                    break

                group2neighbors = d_graph[walk[-1]][neighbors_key]
                all_possible_groups = [group for group in group2neighbors if len(group2neighbors[group]) > 0]
                if len(all_possible_groups) == 0: break
                random_group = np.random.choice(all_possible_groups, size=1)[0]
                walk_options = walk_options[random_group]

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key][random_group]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                    walk.append(walk_to)
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][random_group][walk[-2]]
                    if len(walk_options) != 0:
                        walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                        walk.append(walk_to)

                

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks
