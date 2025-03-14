import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import random
import time
import pickle as pkl
import argparse
from sklearn.metrics import roc_auc_score
from org.gesis.lib import io
from org.gesis.lib.io import create_subfolders
from org.gesis.lib.io import save_csv
from org.gesis.lib.n2v_utils import set_seed, rewiring_list, recommender_model_walker, recommender_model, recommender_model_cw, get_top_recos,get_top_recos_v2, read_graph
from org.gesis.lib.model_utils import get_train_test_graph, get_model_metrics, get_model_metrics_v2
from joblib import delayed
from joblib import Parallel
from collections import Counter
from load_dataset import load_dataset

EPOCHS = 30
main_path = "../AdaptiveAlpha/"

def get_recos_at_random(g, all_nodes):
    recos = list()
    for u in all_nodes:
        candidates = list(nx.non_neighbors(g, u))
        v = np.random.choice(candidates, size=1)[0]
        recos.append((u,v))
    return recos

def make_one_timestep(g, seed,t=0,path="",model="",extra_params=dict()):
        '''Defines each timestep of the simulation:
            0. each node makes experiments
            1. loops in the permutation of nodes choosing the INFLUENCED node u (u->v means u follows v, v can influence u)
            2. loops s (number of interactions times)
            3. choose existing links with 1-a prob, else recommends
                4. if recommendes: invokes recommend_nodes() to choose the influencers nodes that are not already linked u->v

        '''
                              
        # set seed
        set_seed(seed)
        print("Getting Link Recommendations from {} Model ".format(model))
        all_nodes = g.nodes()
        recos = get_recos_at_random(g, all_nodes) 
        print("Recommendations Obtained")
        new_edges = 0
        removed_edges, added_edges = list(), list()
        for i,(u,v) in enumerate(recos):
            seed += i
            set_seed(seed)
            if not g.has_edge(u,v):
               edges_to_be_removed = rewiring_list(g, u, 1)
               removed_edges.extend(edges_to_be_removed)
               added_edges.append((u,v))
               # g.remove_edges_from(edges_to_be_removed) # deleting previously existing links
               new_edges += 1
               # g.add_edge(u,v)
            seed += 1
        g.remove_edges_from(removed_edges)
        g.add_edges_from(added_edges)
        print("No of new edges added: ", new_edges)
        return g


def run(name,model,main_seed,extra_params):
    # try:  
    # Setting seed
    np.random.seed(main_seed)
    random.seed(main_seed)
    folder_path = main_path+"/model_{}_name_{}/seed_{}".format(model,name,main_seed)
    new_filename = get_filename(name,model) +".gpickle"
    new_path = os.path.join(folder_path, new_filename) 
    if os.path.exists(new_path): # disabling this condition
        print("File exists for model: {}, name: {}".format(model,name))
        return 
    

    # Sample testing edges & create training instance g object
    g_train_path = "./data/{}/{}_{}.gpickle".format(name,name,main_seed)
    test_dict_path = "./data/{}/{}_{}_dict.gpickle".format(name,name,main_seed)
    if not os.path.exists(g_train_path) or not os.path.exists(test_dict_path):
        print("Creating training and testing path")
        g = load_dataset(name)     # Initial Graph is read
        print("Total edges in the graph: {} and number of nodes : {} ".format(g.number_of_edges(),g.number_of_nodes()))
        g_train, test_edges, true_labels = get_train_test_graph(g.copy(), main_seed,ds=name)
        io.save_gpickle(g_train, "./data/{}/{}_{}.gpickle".format(name,name,main_seed))
        io.save_pickle({"test_edges":test_edges,"true_labels":true_labels}, "./data/{}/{}_{}_dict.gpickle".format(name,name,main_seed))
    else:
        print("Reading testing and training set from existing paths: ", g_train_path, test_dict_path)
        g_train = io.read_pickle(g_train_path)
        dict_ = io.read_pickle(test_dict_path)
        test_edges, true_labels = dict_["test_edges"], dict_["true_labels"]

    g = g_train

    print("[After sampling] Edges: {} , Nodes: {}".format(g.number_of_edges(),g.number_of_nodes()))
    iterable = tqdm(range(EPOCHS), desc='Timesteps', leave=True) 
    time = 0

    for time in iterable:
        is_file, g_obj =  is_file_exists(name,model,main_seed,time)
        if not is_file:
            print("File does not exist for time {}, creating now".format(time))
            seed = main_seed+time+1 
            g_updated = make_one_timestep(g.copy(),seed,time,new_path,model,extra_params)
           
            g = g_updated
            save_modeldata(g, test_edges,true_labels,name, model,main_seed,t=time)
            save_metadata(g_updated,name, model,main_seed,t=time)
        else:
            print("File exists for time {}, loading it... ".format(time))
            g = g_obj

            if time == EPOCHS-1:
                pass
                    # print("Get graph for utility calculation at time: {}" time)
            
    # except Exception as e:
    #      print("Error in run : ", e)


def is_file_exists(name, model, seed,t):
    folder_path = main_path+"/model_{}_name_{}/seed_{}".format(model,name,seed)
    filename = get_filename(name,model)
    fn = os.path.join(folder_path,'_{}_t_{}.gpickle'.format(filename,t))
    print("checking for existence: ", fn)
    if os.path.exists(fn):
        return True, read_graph(fn)
    else:
        return False, None
    

def get_filename(name, model):
    return "{}-name_{}".format(model,name)

def save_metadata(g, name, model,seed,t=0):
    folder_path = main_path+"/model_{}_name_{}/seed_{}".format(model,name,seed)
    create_subfolders(folder_path)
    filename = get_filename(name,model)
    
    
    fn = os.path.join(folder_path,'_{}_t_{}.gpickle'.format(filename,t))
    io.save_gpickle(g, fn)

    print("Saving graph file at, ", fn.replace(".gpickle",""))


def save_modeldata(g, test_edges, true_labels, name, model,seed,t=0):
        dict_folder = "./utility/model_{}_name_{}/seed_{}".format(model,name,seed)
        if not os.path.exists(dict_folder): os.makedirs(dict_folder)
        dict_file_name = dict_folder+"/_name{}.pkl".format(name)
       
        y_scores = [1 if g.has_edge(u,v) else 0 for (u,v) in test_edges]
        auc_score = roc_auc_score(true_labels,y_scores)
        print("Auc score: {}, for T:{}".format(auc_score,t))
        if not os.path.exists(dict_file_name):
            result_dict = dict()
        else:
            print("Loading pkl file: ", dict_file_name)
            with open(dict_file_name, 'rb') as f:                
                 result_dict = pkl.load(f)
        
        precision, recall = -1, -1 
        result_dict[t] = {"auc_score":auc_score,"precision":precision, "recall":recall}  
        with open(dict_file_name, 'wb') as f:                
            pkl.dump(result_dict,f)
               
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Different Walker Models", default="random", type=str)
    parser.add_argument("--name", help="Real Datasets (rice)", type=str)
    parser.add_argument("--seed", help="Seed", type=int, default=42)

   
    args = parser.parse_args()
    
    start_time = time.time()
    extra_params = dict()
    model = args.model
    run(name=args.name, model=model,main_seed=args.seed,extra_params=extra_params)


    print("--- %s seconds ---" % (time.time() - start_time))
        