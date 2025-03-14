import random
import os
import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import torch
from torchmetrics.functional import pairwise_cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity


from org.gesis.lib.cw_utils import get_upweighted_weights
from org.gesis.lib.io import read_pickle, save_gpickle
from load_dataset import load_rice, load_dataset
from fairwalk.fairwalk  import FairWalk
from deepwalk.deepwalk  import DeepWalker
from node2vec import Node2Vec
from walkers.adaptivealphatest import AdaptiveAlphaTest
from walkers.fastadaptivealphatest import FastAdaptiveAlphaTest
from walkers.fastadaptivealphatestfixed import FastAdaptiveAlphaTestFixed
from walkers.fastfairwalk import FastFairWalk
from walkers.fastcrosswalk import FastCrossWalk

walker_dict = {
"fastadaptivealphatestfixed" : FastAdaptiveAlphaTestFixed,
"ffw": FastFairWalk,
"fcw": FastCrossWalk,
"fastadaptivealphatest" : FastAdaptiveAlphaTest,
}


# Hyperparameter for node2vec/fairwalk
DIM = 64
WALK_LEN = 10
NUM_WALKS = 200

import torch
import dgl
main_path = "../AdaptiveAlpha"

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic =  False
        torch.backends.cudnn.benchmark = False
    
    # dgl.seed(seed)
   #  dgl.random.seed(seed)
    # torch.set_deterministic(True) 
    # torch.use_deterministic_algorithms(True)

def rewiring_list(G, node, number_of_rewiring):
        nodes_to_be_unfollowed = []
        node_neighbors = np.array(list(G.successors(node)))
        nodes_to_be_unfollowed = np.random.permutation(node_neighbors)[:number_of_rewiring]
        return list(map(lambda x: tuple([node, x]), nodes_to_be_unfollowed))

def get_walks(G,model="n2v",extra_params=dict(),num_cores=8):
    if model == "n2v":
         node2vec = Node2Vec(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores)
         return node2vec.walks
    elif model == "fw":
        fw_model = FairWalk(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores)
        return fw_model.walks

    WalkerObj = walker_dict[model] # degree_beta_1.0 for instance
    walkobj = WalkerObj(G, dimensions=DIM, walk_len=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores,**extra_params)
    return walkobj.walks

def recommender_model_walker(G,t=0,model="n2v",extra_params=dict(),num_cores=8, is_walk_viz=False):
    WalkerObj = walker_dict[model.split("_")[0]] # degree_beta_1.0 for instance
    walkobj = WalkerObj(G, dimensions=DIM, walk_len=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores,**extra_params)       
    model = walkobj.fit() 

    # emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()], index = G.nodes))
    embedding_list = np.array([model.wv.get_vector(str(n)) for n in range(G.number_of_nodes())])
    embeddings = torch.from_numpy(embedding_list)
    print("Node embeddings are obtained")
    # embeddings = walkobj.fit()
    # emb_df = (pd.DataFrame([embeddings[n] for n in G.nodes()], index = G.nodes))
    # return model, embeddings
    return None, embeddings


def recommender_model(G,t=0,path="",model="n2v",p=1,q=1,num_cores=8, is_walk_viz=False):
    if model == "n2v":
        print("[N2V] Using p value: {}, Using q value : {}".format(p,q))
        node2vec = Node2Vec(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores,p=p,q=q)
        if is_walk_viz:
          dict_path = path.replace(".gpickle","") + "_frac.pkl"
          get_walk_plots(node2vec.walks, G,t,dict_path)    
        model = node2vec.fit() 
        emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()], index = G.nodes))
    elif model == "fw":
        fw_model = FairWalk(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores,p=p,q=q)
        if is_walk_viz:
            dict_path = path.replace(".gpickle","") + "_frac.pkl"
            print(dict_path)
            get_walk_plots(fw_model.walks, G,t,dict_path)
        model = fw_model.fit() 
        emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()], index = G.nodes))
    # elif model == "ffw":
    #     ffw_model = FastFairWalk(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores,p=p,q=q)
    #     model = ffw_model.fit() 
    #     emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()], index = G.nodes))
    return model, emb_df

def recommender_model_cw(G, t=0, path="", p=1, alpha=0.5, num_cores=8):

    print("[CrossWalk] Using p value: {}, Using alpha value : {}".format(p,alpha))
    print("#1 -> Generate Random Walks")
    deepwalk = DeepWalker(G, dimensions=DIM, walk_len=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores)    
    walks = deepwalk.walks
    print("#2 -> Upweight the edges")
    weight_dict = get_upweighted_weights(G, walks, p, alpha, d=WALK_LEN, r=NUM_WALKS)
    print("#3 -> Update with weight attrs into G")
    nx.set_edge_attributes(G, weight_dict)
    print("#4 -> Generate node embeddings")
    cw_model = DeepWalker(G, dimensions=DIM, walk_len=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores) 
    
    model = cw_model.fit() 
    emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()], index = G.nodes))
    return model, emb_df

def get_top_recos(g, embeddings, u, N=1):
    all_nodes = g.nodes()
    df = embeddings
    results = []
    for src_node in u:
        source_emb = df[df.index == src_node]
        other_nodes = [n for n in all_nodes if n not in list(g.adj[src_node]) + [src_node]]
        other_embs = df[df.index.isin(other_nodes)]

        sim = cosine_similarity(source_emb, other_embs)[0].tolist()
        idx = other_embs.index.tolist()

        idx_sim = dict(zip(idx, sim))
        idx_sim = sorted(idx_sim.items(), key=lambda x: x[1], reverse=True)
        
        similar_nodes = idx_sim[:N]
        v = [tgt[0] for tgt in similar_nodes][0]
        results.append((src_node,v))
    print("len(results) in recos method: ", len(results))   
    return results 


def get_top_recos_v2(g, embeddings, all_nodes, N=1):
    cosine_sim = pairwise_cosine_similarity(embeddings, embeddings)
    print("Obtained Cosine Similarity Values")
    results = []
    print("Finding top:{} recos based on cos-sim".format(N))
    adj_matrix =  nx.to_numpy_array(g, nodelist=list(range(g.number_of_nodes()))).astype(np.float32)
    np.fill_diagonal(adj_matrix, 1)
    print("creating a torch matrix")

    MAX_LIMIT = 5000
    start = 0
    end = len(all_nodes)
    while True:
        if start >= end: break
        end_lim = start + MAX_LIMIT
        if end_lim >= end: end_lim = end
        print("Spanning nodes from : {} to  {}".format(start, end_lim))
        cosine_sim_sub = cosine_sim[start:end_lim, :]
      
        with torch.no_grad():
            adj_matrix_torch = torch.tensor(adj_matrix[start:end_lim, :])
        adj_matrix_torch[adj_matrix_torch == 1.0] = float(-1000)
        adj_matrix_torch += cosine_sim_sub
       
        _, tgt_nodes = torch.topk(adj_matrix_torch, N, dim=1, sorted=False)

        src_nodes = list(range(start, end_lim, 1))
        for src_node, tgts in zip(src_nodes, tgt_nodes):
            results.extend([(src_node, int(tgt)) for tgt in tgts])
    
        
        start += MAX_LIMIT
 
    #         # adj_matrix = torch.tensor(adj_matrix, device="cpu")
    #         # new_diags = adj_matrix.diagonal() + 1.0
    #         # adj_matrix.diagonal().copy_(new_diags)
    #         # del new_diags
    #         # adj_matrix[adj_matrix == 1.0] = float(-1000)
    #         # adj_matrix += cosine_sim
    #         # del cosine_sim
    #         # _, tgt_nodes = torch.topk(adj_matrix, N, dim=1)
    #         # del adj_matrix
    #         # for i, tgts in enumerate(tgt_nodes):
    #         #      results.extend([(i,int(tgt)) for tgt in tgts])
 
    # for src_node in all_nodes:
       
    #     other_nodes = torch.tensor([n for n in all_nodes if n not in list(g.adj[src_node]) + [src_node]])
    #     sims = cosine_sim[src_node, other_nodes]
    #     _, tgt_idxs = torch.topk(sims, N)
    #     tgt_nodes = other_nodes[tgt_idxs]

    #     for tgt in tgt_nodes:
    #          results.append((src_node, int(tgt)))

    print("len(results) in recos method: ", len(results))   
    return results, cosine_sim

def get_walk_plots(walks, g, t,dict_path):
    try:
        print("get trace for t = ", t)
        node_to_walk_dict = {}
        len_majority_nodes  = len([node for node in g.nodes() if g.nodes[node]["m"] == 0])
        len_minority_nodes = len(g.nodes()) - len_majority_nodes
        
        for sub_walk in walks:
         
            sub_walk = [int(node) for node in sub_walk]
            src_node = sub_walk[0]
            if src_node not in node_to_walk_dict:
                node_to_walk_dict[src_node] = list(set(sub_walk))
            else:
                new_list = list(set(node_to_walk_dict[src_node]+sub_walk))
                node_to_walk_dict[src_node] = new_list
        
        maj_frac, min_frac = 0, 0
       
    
        for _, walk in node_to_walk_dict.items():
           
            majority_nodes_visited = len([_ for node in walk if g.nodes[node]["m"] == 0])
            minority_nodes_visited = len([_ for node in walk if g.nodes[node]["m"] == 1])
            maj_frac += round(majority_nodes_visited/len_majority_nodes,2)
            min_frac += round(minority_nodes_visited/len_minority_nodes,2)
            
        avg_maj_frac = round(maj_frac/len(g.nodes()),2)
        avg_min_frac = round(min_frac/len(g.nodes()),2)
        print("~~~~~~~~~~~ maj frac: {}, min frac:{}".format(avg_maj_frac, avg_min_frac))
        if os.path.exists(dict_path):
             with open(dict_path, 'rb') as f:
                dict_ = pkl.load(f)
        else:
            dict_ = {}

        # also compute avg betweenness centrality
        centrality_dict = nx.betweenness_centrality(g, normalized=True)
        minority_centrality = [val for node, val in centrality_dict.items() if g.nodes[node]["m"] == 1]
        avg_val = np.mean(minority_centrality)    
        
        dict_[t] = {"maj":avg_maj_frac, "min": avg_min_frac,"bet":avg_val}
        print("!! comes here to write")
        os.makedirs(os.path.dirname(dict_path), exist_ok=True)
        with open(dict_path, 'wb') as f:
           pkl.dump(dict_, f)
       
    except Exception as error:
        print("Error in get walk plots: ", error)

def get_diff_group_centrality(g,centrality_dict,group):
    

    node_attrs = nx.get_node_attributes(g,"group")

    centrality_1 = [val for node, val in centrality_dict.items() if node_attrs[node] == group]
    avg_val_1 = np.mean(centrality_1)

    centrality_2 = [val for node, val in centrality_dict.items() if node_attrs[node] != group]
    avg_val_2 = np.mean(centrality_2)
    return avg_val_1 - avg_val_2
 
def get_avg_group_centrality(g,centrality_dict,group=1):
    

    node_attrs = nx.get_node_attributes(g,"group")

    centrality = [val for node, val in centrality_dict.items() if node_attrs[node] == group]
    avg_val = np.mean(centrality)
    return avg_val

def read_graph(file_name,seed=None):
    if "baseline" in file_name and "rice" in file_name:
        name = "rice"
        dsname = "./data/{}/{}_{}.gpickle".format(name,name,seed)
        g = read_pickle(dsname)
    elif "baseline" in file_name and "twitter" in file_name:
        name = "twitter"
        dsname = "./data/{}/{}_{}.gpickle".format(name,name,seed)
        g = read_pickle(dsname)
    elif "baseline" in file_name and "synth2" in file_name:
        name = "synth2"
        dsname = "./data/{}/{}_{}.gpickle".format(name,name,seed)
        g = read_pickle(dsname)
    elif "baseline" in file_name and "tuenti" in file_name:
        name = "tuenti"
        dsname = "./data/{}/{}_{}.gpickle".format(name,name,seed)
        g = read_pickle(dsname)
    elif "baseline" in file_name and "pokec" in file_name:
        name = "pokec"
        dsname = "./data/{}/{}_{}.gpickle".format(name,name,seed)
        g = read_pickle(dsname)
    elif "baseline" in file_name and "facebook" in file_name:
        name = "facebook"
        dsname = "./data/{}/{}_{}.gpickle".format(name,name,seed)
        g = read_pickle(dsname)
    else:
        with open(os.path.join(file_name), 'rb') as f:
                g = pkl.load(f)
    try:
        node2group = {node:g.nodes[node]["m"] for node in g.nodes()}
        nx.set_node_attributes(g, node2group, 'group')
    except Exception as e:
        pass
        # print("This should be a real graph. Group attributes should be already set.")

    return g

def get_centrality_dict(model,g,hMM,hmm,centrality="betweenness"):        
    dict_folder = "./centrality/{}/{}".format(centrality,model+"_fm_0.3")
    print("Dict folder: ", dict_folder)
    if not os.path.exists(dict_folder): os.makedirs(dict_folder)
    dict_file_name = dict_folder+"/_hMM{}_hmm{}.pkl".format(hMM,hmm)
        
    if not os.path.exists(dict_file_name):
            if centrality == "betweenness":
                centrality_dict = nx.betweenness_centrality(g, normalized=True)
            elif centrality == "closeness":
                centrality_dict = nx.closeness_centrality(g)
            else:
                print("Invalid Centrality measure")
                return
            print("Generating pkl file: ", dict_file_name)
            with open(dict_file_name, 'wb') as f:                
                pkl.dump(centrality_dict,f)
    else:
            print("Loading pkl file: ", dict_file_name)
            with open(dict_file_name, 'rb') as f:                
                centrality_dict = pkl.load(f)
    return centrality_dict
