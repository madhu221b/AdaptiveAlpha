import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import random
import time
import pickle as pkl
import argparse

from org.gesis.lib import io
from org.gesis.lib.io import create_subfolders
from org.gesis.lib.io import save_csv
from org.gesis.lib.n2v_utils import set_seed, rewiring_list, recommender_model_walker, recommender_model, recommender_model_cw, get_top_recos,get_top_recos_v2, read_graph
from org.gesis.lib.model_utils import get_train_test_graph, get_model_metrics, get_model_metrics_v2, get_disparity
from joblib import delayed
from joblib import Parallel
from collections import Counter
from load_dataset import load_dataset

EPOCHS = 30
main_path = "../AdaptiveAlpha/"
   
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

        print("Generating Node Embeddings")
        # if "ffw" in model:
        #     p, q = extra_params["p"], extra_params["q"]
        #     _, embeds = recommender_model(g,t,path,model="ffw",p=p,q=q)
        if "fw" in model and "ffw" not in model:
            p, q = extra_params["p"], extra_params["q"]
            _, embeds = recommender_model(g,t,path,model="fw",p=p,q=q)
        elif "n2v" in model:
            p, q = extra_params["p"], extra_params["q"]
            _, embeds = recommender_model(g,t,path,model="n2v",p=p,q=q)
        # elif args.model in ["cw"]:
        #      p_cw, alpha_cw = extra_params["p_cw"], extra_params["alpha_cw"]
        #      _, embeds = recommender_model_cw(g, t, path, p=p_cw, alpha=alpha_cw)
        else:
            _, embeds = recommender_model_walker(g,t,model=model,extra_params=extra_params)
        print("Getting Link Recommendations from {} Model ".format(model))
        all_nodes = g.nodes()
        recos, cossim = get_top_recos_v2(g,embeds, all_nodes) 
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
        return g, embeds, cossim


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
            g_updated, embeds, cossim = make_one_timestep(g.copy(),seed,time,new_path,model,extra_params)
           
            g = g_updated
            save_modeldata(embeds, cossim, test_edges, true_labels,name, model,main_seed,t=time)
            save_metadata(g_updated,name, model,main_seed,t=time)
            # get_disparity(g,cossim,test_edges,true_labels)
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


def save_modeldata(embeds,cossim,test_edges, true_labels, name, model,seed,t=0):
        dict_folder = "./utility/model_{}_name_{}/seed_{}".format(model,name,seed)
        if not os.path.exists(dict_folder): os.makedirs(dict_folder)
        dict_file_name = dict_folder+"/_name{}.pkl".format(name)

        # precision, recall = get_model_metrics(g,test_edges,true_labels)
        auc_score, precision, recall = get_model_metrics_v2(embeds,cossim,test_edges,true_labels)
        # print("Recall: {}, Precision: {} for hMM:{}, hmm:{} for T={}".format(recall, precision, hMM, hmm,t))
        print("Auc score: {}, for T:{}".format(auc_score,t))
        if not os.path.exists(dict_file_name):
            result_dict = dict()
        else:
            print("Loading pkl file: ", dict_file_name)
            with open(dict_file_name, 'rb') as f:                
                 result_dict = pkl.load(f)
        
        # result_dict[t] = {"precision":precision, "recall":recall}
        result_dict[t] = {"auc_score":auc_score,"precision":precision, "recall":recall}  
        with open(dict_file_name, 'wb') as f:                
            pkl.dump(result_dict,f)
               
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Different Walker Models", type=str)
    parser.add_argument("--name", help="Real Datasets (rice)", type=str)
    parser.add_argument("--p", help="Return parameter", type=float, default=1.0)
    parser.add_argument("--q", help="In-out parameter", type=float, default=1.0)
    parser.add_argument("--beta", help="Beta paramater", type=float, default=2.0)
    parser.add_argument("--alpha", help="Alpha paramater (Levy)", type=float, default=1.0)
    parser.add_argument("--seed", help="Seed", type=int, default=42)

    parser.add_argument("--p_cw", help="[CrossWalk] Degree of biasness of random walks towards visiting nodes at group boundaries", type=float, default=2)
    parser.add_argument("--alpha_cw", help="[CrossWalk] Upweights edges connecting different groups [0,1]", type=float, default=0.5)

   
    args = parser.parse_args()
    
    start_time = time.time()
    extra_params = dict()
    if args.model in ["fw","ffw","n2v"]:
        model = args.model + "_p_{}_q_{}".format(args.p,args.q)
        extra_params = {"p":args.p,"q":args.q}
    elif args.model in ["fcw"]:
        model = args.model + "_p_{}_alpha_{}".format(args.p_cw,args.alpha_cw)
        extra_params = {"p":args.p_cw,"alpha":args.alpha_cw}
    elif args.model in  ["nlindlocalind","fastadaptivealphatestfixed"]:
        model = "{}_alpha_{}_beta_{}".format(args.model,args.alpha,args.beta)
        extra_params = {"alpha":args.alpha,"beta":args.beta}
    else:
       model =  "{}_beta_{}".format(args.model,args.beta)
       extra_params = {"beta":args.beta}

    run(name=args.name, model=model,main_seed=args.seed,extra_params=extra_params)


    print("--- %s seconds ---" % (time.time() - start_time))
        