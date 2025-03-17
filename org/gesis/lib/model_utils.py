import os
import numpy as np
import networkx as nx
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
import pickle as pkl

import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import scipy.sparse as sp

from dgl import from_networkx
from dgl.sampling import  global_uniform_negative_sampling

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    os.environ.__setitem__('DISPLAY', ':0.0')
    mpl.use('Agg')

def get_edge_dict(g,edges=list(),is_neg=False):
    edge_dict = dict()
    node_attr = nx.get_node_attributes(g, "group")
    
    if is_neg is False: edges = g.edges()
    
 
    for u, v in edges:
        u, v = int(u), int(v)
        key = "{}->{}".format(node_attr[u],node_attr[v])    
        if key not in edge_dict:
            edge_dict[key] = [(u,v)]
        else:
            edge_dict[key].append((u,v))
    return edge_dict


def generate_pos_neg_links(g,seed, prop_pos=0.1, prop_neg=0.1, ds="rice"):
        """

        Following CrossWalk's methodology to sample test edges

        Select random existing edges in the graph to be postive links,
        and random non-edges to be negative links.
        prop_pos: 0.1,  # Proportion of edges to remove and use as positive samples per edge group
        prop_neg: 0.1  # Number of non-edges to use as negative samples per edge group
        """
        _rnd = np.random.RandomState(seed=seed)
        pos_edge_list, neg_edge_list = [], []

        # Select n edges at random (positive samples)
        n_edges = g.number_of_edges()
        n_nodes = g.number_of_nodes()
 

        
        if ds in ["tuenti","pokec"]:
            dgl_g = from_networkx(g)
            u, v = global_uniform_negative_sampling(dgl_g,num_samples=g.number_of_edges())
            u, v = u.unsqueeze(1), v.unsqueeze(1)
            non_edges = torch.hstack((u,v))
        else:
            non_edges = list(nx.non_edges(g))
    
        
        pos_edge_dict = get_edge_dict(g)
        neg_edge_dict = get_edge_dict(g,non_edges,is_neg=True)


        for edge_type, edges in pos_edge_dict.items():
            pos_edges, neg_edges = edges, neg_edge_dict[edge_type]

            pos_n_edges, neg_n_edges = len(pos_edges), len(neg_edges)
            npos =  int(prop_pos*pos_n_edges)
            nneg = int(prop_neg*neg_n_edges)
            print("Edge Type: {} , total edges: {}, sampling pos links: {} , total neg edges: {}, sampling neg links: {} ".format(edge_type,pos_n_edges,npos,neg_n_edges,nneg))
            rnd_pos_inx = _rnd.choice(pos_n_edges, npos, replace=False)
            pos_edge_list.extend([edges[ii] for ii in rnd_pos_inx])


            rnd_neg_inx = _rnd.choice(neg_n_edges, nneg, replace=False)
            neg_edge_list.extend([neg_edges[ii] for ii in rnd_neg_inx])
        
        pos_edge_list, neg_edge_list = list(set(pos_edge_list)), list(set(neg_edge_list))
        print("Totally pos set: {}, total neg set: {}".format(len(pos_edge_list),len(neg_edge_list)))
        return pos_edge_list, neg_edge_list

def get_train_test_graph(g, seed, ds):
    """
    Input is graph read at t=0 (DPAH graph)
    
    Return training graph instance and list of pos-neg test edges, and true labels
    """
    print("pre split nx.isolates(g): ",  len(list(nx.isolates(g))), g.number_of_nodes())
    pos_edge_list, neg_edge_list = generate_pos_neg_links(g,seed,prop_pos=0.1,prop_neg=0.1, ds=ds)
    g.remove_edges_from(pos_edge_list)
    print("~~~~~post split nx.isolates(g): ", len(list(nx.isolates(g))), g.number_of_nodes())
    edges = pos_edge_list + neg_edge_list
    labels = np.zeros(len(edges))
    labels[:len(pos_edge_list)] = 1
    return g, edges, labels

def get_cos_sims(df, test_edges):
    scores = list()
    for u,v  in test_edges:
        source_emb = df[df.index == u]
        target_emb = df[df.index == v]
      
        sim = cosine_similarity(source_emb, target_emb)[0]
        scores.append(sim)
           
    return scores


def get_model_metrics(g,test_edges,y_true):
    """
    Computes Precision & Recall
    - Precision: Quantifies the number of correct positive predictions made.
       Ratio of correctly predicted positive examples divided by the total number of positive examples that were predicted.
     
    - Recall: Calculated as the number of true positives divided by the total number of true positives and false negatives. 

    
    """
    y_pred = [int(g.has_edge(u,v)) for u,v in test_edges]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("tn: {},  fp:{},  fn: {},  tp: {}".format(tn, fp, fn, tp))
    return precision, recall, accuracy

def get_model_metrics_v2(sim_matrix, test_edges, y_true):
    """
    Computes Precision & Recall
    - Precision: Quantifies the number of correct positive predictions made.
       Ratio of correctly predicted positive examples divided by the total number of positive examples that were predicted.
     
    - Recall: Calculated as the number of true positives divided by the total number of true positives and false negatives. 

    
    """
    print("Calculating auc scores for test edges")
    y_pred = list()
    for (u,v) in test_edges:
        val = sim_matrix[u,v]
        y_pred.append(float(val))

    # y_pred = get_cos_sims(embeddings,test_edges)
    auc_score = roc_auc_score(y_true,y_pred)
    print("auc score: ", auc_score)
    

    y_scores = y_pred
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Threshold value is:", optimal_threshold)

    # accuracy
    threshold = optimal_threshold
    y_pred_vals = np.where(np.array(y_pred) >= threshold,1,0)
    accuracy = accuracy_score(y_true, y_pred_vals)
    precision = precision_score(y_true, y_pred_vals)
    recall = recall_score(y_true, y_pred_vals)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_vals).ravel()
    print("tn: {},  fp:{},  fn: {},  tp: {}".format(tn, fp, fn, tp))
    print("accuracy: ", accuracy, "precision: ", precision, "recall: ", recall)


    return auc_score, precision, recall

def get_disparity(graph,cos_sim,test_edges,y_true):
    """
    Computes Precision & Recall
    - Precision: Quantifies the number of correct positive predictions made.
       Ratio of correctly predicted positive examples divided by the total number of positive examples that were predicted.
     
    - Recall: Calculated as the number of true positives divided by the total number of true positives and false negatives. 

    
    """
    print("Calculating auc scores for test edges")
    y_pred = list()
    node_attr = nx.get_node_attributes(graph, "group")
    group2scores = dict()
    for (u,v), true_label in zip(test_edges,y_true):
        val = cos_sim[u,v]
        y_pred.append(float(val))
        key = (node_attr[u],node_attr[v])
        if key not in group2scores: group2scores[key] = {"pred": list(), "true":list()}
        group2scores[key]["pred"].append(float(val))
        group2scores[key]["true"].append(true_label)


    # y_pred = get_cos_sims(embeddings,test_edges)
    auc_score = roc_auc_score(y_true,y_pred)
    print("auc score: ", auc_score)
    

    y_scores = y_pred
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Threshold value is:", optimal_threshold)

    # accuracy
    threshold = optimal_threshold
    y_pred_vals = np.where(np.array(y_pred) >= threshold,1,0)
    accuracy = accuracy_score(y_true, y_pred_vals)
    precision = precision_score(y_true, y_pred_vals)
    recall = recall_score(y_true, y_pred_vals)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_vals).ravel()
    print("tn: {},  fp:{},  fn: {},  tp: {}".format(tn, fp, fn, tp))
    print("accuracy: ", accuracy, "precision: ", precision, "recall: ", recall)
   
    var_arr = list()
    for group, scores in group2scores.items():
        count = 0
        for y_pred, y_true in zip(scores["pred"],scores["true"]):
            if y_pred >= threshold and y_true:
                count += 1
        print("count: {}, total: {}".format(count,len(scores["pred"])))
        count = count/len(scores["pred"])

        var_arr.append(count)
    
    print("var arr: ", var_arr)
    acc_dis = np.var(var_arr)*100.0
    print("accuracy disparity: {}, for {} groups".format(acc_dis,len(var_arr)))
    print("accuracy: ", accuracy*100.0)
    return auc_score, precision, recall