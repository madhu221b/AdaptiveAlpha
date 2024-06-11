
import numpy as np
import networkx as nx
import pandas as pd
import time

from degreewalker import DegreeWalker
from indegreewalker import InDegreeWalker
from commonnghaware import CommonNeighborWalker
from levywalker import LevyWalker
from fairindegreewalker import FairInDegreeWalker
from indegreevarybetawalker import InDegreeVaryBetaWalker
from indegreevarybetawalkerv2 import InDegreeVaryBetaWalkerV2
from ingroupdegreewalker import InGroupDegreeWalker
from highlowindegreewalker import HighLowInDegreeWalker
# from nonlocalindegreewalker import NonLocalInDegreeWalker
from nonlocaladaptivealpaindegreewalker import NonLocalAdaptiveInDegreeWalker
from nonlocalindegreelocalrandomwalker import NonLocalInDegreeLocalRandomWalker
from nllindegreelocalrandomwalker import NllInDegreeLocalRandomWalker
from nonlocalindlocalindwalker import NonLocalInDegreeLocalInDegreeWalker
from beepboopv3_walker import NonLocalAdaptiveInDegreeLocalRandomWalkerBeepBoopV3

# path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/DPAH_trial/DPAH-N100-fm0.3-d0.03-ploM2.5-plom2.5-hMM0.8-hmm0.8-ID0.gpickle"
path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/DPAH_fm_0.3/DPAH-N20-fm0.3-d0.03-ploM2.5-plom2.5-hMM0.2-hmm0.8-ID0.gpickle"

# path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/DPAH_trial/DPAH-N10-fm0.3-d0.03-ploM2.5-plom2.5-hMM0.8-hmm0.2-ID0.gpickle"
graph = nx.read_gpickle(path)
print(graph)
node2group = {node:graph.nodes[node]["m"] for node in graph.nodes()}
nx.set_node_attributes(graph, node2group, 'group')

DIM = 64
WALK_LEN = 10
NUM_WALKS = 200

start_time = time.time()
# walker = InDegreeVaryBetaWalker(graph, dimensions=DIM, walk_len=WALK_LEN, num_walks=NUM_WALKS, workers=8)
# walker = LevyWalker(graph,alpha=1,workers=8,dimensions=64)
# walker = FairInDegreeWalker(graph, beta=2, workers=8,dimensions=64)
# walker = InGroupDegreeWalker(graph, dimensions=DIM, walk_len=WALK_LEN, num_walks=NUM_WALKS, workers=8)

walker = InDegreeVaryBetaWalkerV2(graph, beta=2, dimensions=DIM, walk_len=WALK_LEN, num_walks=NUM_WALKS, workers=8)
# walker = InDegreeWalker(graph, beta=2, dimensions=DIM, walk_len=WALK_LEN, num_walks=NUM_WALKS, workers=8)
# walker = HighLowInDegreeWalker(graph, alpha=0.5, dimensions=DIM, walk_len=WALK_LEN, num_walks=NUM_WALKS, workers=8)
# walker = NonLocalInDegreeLocalInDegreeWalker(graph,beta=2,alpha=1.0, dimensions=DIM, walk_len=WALK_LEN, num_walks=NUM_WALKS, workers=8)
# walker = NonLocalAdaptiveInDegreeWalker(graph,beta=2,dimensions=DIM, walk_len=WALK_LEN, num_walks=NUM_WALKS, workers=8)
# walker = NonLocalAdaptiveInDegreeLocalRandomWalkerBeepBoopV3(graph,beta=2,dimensions=DIM, walk_len=WALK_LEN, num_walks=NUM_WALKS, workers=8)
model = walker.fit() 
# print("model walks:", walker.walks)
end_time = time.time()

print("Time taken: ", (end_time-start_time))

emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in graph.nodes()], index = graph.nodes))
print("embedding shape: ", emb_df.shape)