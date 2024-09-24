import os
import networkx as nx
import numpy as np
from org.gesis.lib.n2v_utils import read_graph
from org.gesis.lib.io import read_pickle

MAIN_PATH = "../AdaptiveAlpha/"
T = 30


def get_edge_dict(recos, g):
    count_dict = dict()
    reco_dict = dict()
    node_attr = nx.get_node_attributes(g, "group")
    for u,v in g.edges():
        u, v = node_attr[u], node_attr[v]
        if (u,v) not in count_dict: count_dict[(u,v)] = 0
        count_dict[(u,v)] += 1
    
    for u,v in recos:
        u, v = node_attr[u], node_attr[v]
        if (u,v) not in reco_dict: reco_dict[(u,v)] = 0
        reco_dict[(u,v)] += 1
    return reco_dict, count_dict

def get_statistical_imparity(model, use_syn_ds=False, hMM=0.0, hmm=0.0, ds=None):
    mean_sp = list()
    for t in range(0,T):

        if use_syn_ds:
            if t == 0:
                prev_gpath = MAIN_PATH+"DPAH_fm_0.3/DPAH-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}-ID0.gpickle".format(hMM,hmm)
            else: 
                prev_gpath = os.path.join(MAIN_PATH,"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,t-1))

            curr_gpath = os.path.join(MAIN_PATH,"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,t))
        else: # use real dataset
            seed = 42
            if t == 0:
                prev_gpath = MAIN_PATH+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format("baseline",ds,seed,"baseline",ds,t-1)
            else: 
                prev_gpath = MAIN_PATH+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t-1)

            curr_gpath = MAIN_PATH+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t)
       
        g_prev = read_graph(prev_gpath,seed=seed)
        g_curr = read_graph(curr_gpath,seed=seed)

        recos = list(set(g_curr.edges())-set(g_prev.edges())) # get the total recommendations
        reco_dict, edge_dict = get_edge_dict(recos,g_curr)

        p_ijs = [reco_dict.get(k,0)/v for k, v in edge_dict.items()]
        var_sp = np.var(p_ijs)
        mean_sp.append(var_sp)

    total_sp = np.mean(mean_sp)
    print("Model - {} Statistical Imparity for T = {} is {:2e}".format(model,T,total_sp))
    return total_sp


def get_er_nwlevel(model, use_syn_ds=False, hMM=0.0, hmm=0.0, ds=None):
    mean_erg = list()
    for t in range(0,T):

        if use_syn_ds:
            if t == 0:
                prev_gpath = MAIN_PATH+"DPAH_fm_0.3/DPAH-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}-ID0.gpickle".format(hMM,hmm)
            else: 
                prev_gpath = os.path.join(MAIN_PATH,"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,t-1))

            curr_gpath = os.path.join(MAIN_PATH,"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,t))
        else: # use real dataset
            seed = 42 # use a fixed seed 
            if t == 0:
                prev_gpath = MAIN_PATH+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format("baseline",ds,seed,"baseline",ds,t-1)
            else: 
                prev_gpath = MAIN_PATH+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t-1)

            curr_gpath = MAIN_PATH+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t)
       
        g_prev = read_graph(prev_gpath,seed=seed)
        g_curr = read_graph(curr_gpath,seed=seed)

        recos = list(set(g_curr.edges())-set(g_prev.edges())) # get the total recommendations
        assert len(recos) == g_curr.number_of_nodes()
        reco_dict, _ = get_edge_dict(recos,g_curr)

        var_erg = np.var(list(reco_dict.values()))
        mean_erg.append(var_erg)

    total_erg = np.mean(mean_erg)
    print("Model - {} Equality of Rep at Network Level for T = {} is {}".format(model,T,total_erg))
    return total_erg

def get_er_userlevel(model, use_syn_ds=False, hMM=0.0, hmm=0.0, ds=None):
    ### DOUBT ###
    seed = 42 # use a fixed seed 
    mean_eru = dict()
    for t in range(0,T):

        if use_syn_ds:
            if t == 0:
                prev_gpath = MAIN_PATH+"DPAH_fm_0.3/DPAH-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}-ID0.gpickle".format(hMM,hmm)
            else: 
                prev_gpath = os.path.join(MAIN_PATH,"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,t-1))

            curr_gpath = os.path.join(MAIN_PATH,"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,t))
        else: # use real dataset
            if t == 0:
                prev_gpath = MAIN_PATH+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format("baseline",ds,seed,"baseline",ds,t-1)
            else: 
                prev_gpath = MAIN_PATH+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t-1)

            curr_gpath = MAIN_PATH+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t)
       
        g_prev = read_graph(prev_gpath,seed=seed)
        g_curr = read_graph(curr_gpath,seed=seed)

        recos = list(set(g_curr.edges())-set(g_prev.edges())) # get the total recommendations
        assert len(recos) == g_curr.number_of_nodes()
        
        node_attr = nx.get_node_attributes(g_curr,"group")
        unique_zs = set(node_attr.values())
        for z in unique_zs:
            num = len([v for _, v in recos if node_attr[v] == z])
            den = g_curr.number_of_nodes()
            bias_erz = num/den
            if z not in mean_eru: mean_eru[z] = list()
            mean_eru[z].append(bias_erz)
            


    for zs, v_list in mean_eru.items():
        assert len(v_list) == T  
        mean_val = 1/len(mean_eru.keys()) - np.mean(v_list)
        print("Model - {} Equality of User Level for Attr:{} at User Level for T = {} is {}".format(model,zs,T,mean_val))


def get_disparity(model, use_syn_ds=False, hMM=0.0, hmm=0.0, ds=None):

    reco_list = list()
    seed = 42 # use a fixed seed 
    for t in range(0,T):

        if use_syn_ds:
            if t == 0:
                prev_gpath = MAIN_PATH+"DPAH_fm_0.3/DPAH-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}-ID0.gpickle".format(hMM,hmm)
            else: 
                prev_gpath = os.path.join(MAIN_PATH,"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,t-1))

            curr_gpath = os.path.join(MAIN_PATH,"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,t))
        else: # use real dataset
            if t == 0:
                prev_gpath = MAIN_PATH+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format("baseline",ds,seed,"baseline",ds,t-1)
            else: 
                prev_gpath = MAIN_PATH+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t-1)

            curr_gpath = MAIN_PATH+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t)
       
        g_prev = read_graph(prev_gpath,seed=seed)
        g_curr = read_graph(curr_gpath,seed=seed)

        recos = list(set(g_curr.edges())-set(g_prev.edges())) # get the total recommendations
        reco_list.extend(recos)
        assert len(recos) == g_curr.number_of_nodes()
                
        node_attr = nx.get_node_attributes(g_curr,"group")
        
    reco_list = set(reco_list)

    true_edges = list()
    true_edge_dict = read_pickle("./data/{}/{}_{}_dict.gpickle".format(ds,ds,seed))       
    test_edges, true_labels = true_edge_dict["test_edges"], true_edge_dict["true_labels"]
    for test_edge, true_label in zip(test_edges,true_labels):
        if true_label == 1: true_edges.append(test_edge)

    total_true, reco_true = dict(), dict()
    for u,v in true_edges:
        id_u, id_v = node_attr[u], node_attr[v]
        if (id_u,id_v) not in total_true: total_true[(id_u,id_v)] = 0
        if  (id_u,id_v) not in reco_true: reco_true[(id_u,id_v)] = 0
        total_true[(id_u,id_v)] += 1
        if (u,v) in reco_list: reco_true[(id_u,id_v)] += 1
    
    final_list = list()
    for k, v in total_true.items():
        val = reco_true[k]/v
        final_list.append(val)
    
    final_var = np.var(final_list)
    print("Model - {} Disparity - {}".format(model,final_var))


  