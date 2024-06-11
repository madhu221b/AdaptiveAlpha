import os
import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import operator

# import matplotlib
# matplotlib.use('Agg')
import matplotlib as mpl


label_dict = {
"nlindlocalind_alpha_1.0_beta_2.0": r"$\alpha=1$", 
"indegreevarybetav2_beta_2.0": "Adaptive " + r"$\alpha$", # newest,
"n2v_p_1.0_q_1.0": "n2v",
"fw_p_1.0_q_1.0": "Fairwalk"
}

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    os.environ.__setitem__('DISPLAY', ':0.0')
    mpl.use('Agg')

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

from org.gesis.lib.n2v_utils import get_walks, get_avg_group_centrality, read_graph, get_centrality_dict, get_diff_group_centrality, avg_centrality
from generate_heatmap_centrality import get_grid

T = 30
hMM_list, hmm_list = np.arange(0,1.1,0.1), np.arange(0,1.1,0.1)



def print_visibility(file_name):
 
    if not os.path.exists(file_name): 
        print("File name does not exist: ", file_name)
        path2 = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/model_n2v_nohuman_p_1.0_q_1.0_name_rice/seed_42/_n2v_p_1.0_q_1.0-name_rice_t_29.gpickle"
        print(path2, os.path.exists(path2))

        return dict()
    g = read_graph(file_name)
    vis_dict = dict() 
    groups = list(set(nx.get_node_attributes(g, "group").values()))
    print("groups:", groups)
    
    # elif "twitter" in file_name: min_grp, maj_grp = 2, 1
    cent_file = file_name.replace(".gpickle","") + ".pkl"
    if not os.path.exists(cent_file):
        centrality_dict = nx.betweenness_centrality(g, normalized=True)
        print("Generating pkl file: ", cent_file)
        with open(cent_file, 'wb') as f:                
                pkl.dump(centrality_dict,f)
    else:
        print("Loading cent file: ", cent_file)
        with open(cent_file,"rb") as f:
              centrality_dict = pkl.load(f)
    # print(centrality_dict)
    for group in groups:
        avg_bet = get_avg_group_centrality(g,centrality_dict,group=group)
        vis_dict[group] = avg_bet
    print("vis dict returned: ", vis_dict)
    return vis_dict

def print_fairness(file_name):
 
    if not os.path.exists(file_name): 
        print(file_name)
        return dict()
    g = read_graph(file_name)
    fairness_dict = dict() 
    groups = list(set(nx.get_node_attributes(g, "group").values()))
    

    cent_file = file_name.replace(".gpickle","") + ".pkl"
    if not os.path.exists(cent_file):
         centrality_dict = nx.betweenness_centrality(g, normalized=True)
         print("Generating pkl file: ", cent_file)
         with open(cent_file, 'wb') as f:                
                pkl.dump(centrality_dict,f)
    else:
        print("Loading cent file: ", cent_file)
        with open(cent_file,"rb") as f:
              centrality_dict = pkl.load(f)

    for group in groups:
        avg_diff =  get_diff_group_centrality(g,centrality_dict,group=group)
        fairness_dict[group] = avg_diff
    return fairness_dict

def print_utility(file_name):
    prec, recall, acc, auc = 0, 0, 0, 0
    try:
  
        if os.path.exists(file_name):
            with open(file_name,"rb") as f:
                result_dict = pkl.load(f)
       
            prec, recall, acc = result_dict[29].get("precision",0), result_dict[29].get("recall",0), result_dict[29].get("accuracy",0)
            auc = result_dict[29].get("auc_score",0)
        else:
            prec, recall, acc, auc = 0, 0, 0, 0
    except Exception as e:
        print("error in print utility: ", e)
    return prec, recall, acc, auc

def count_edges(walks):
    count_dict = dict()
    topk = 10

    for walk in walks:
        walk = [int(_) for _ in walk]
        if len(walk) == 1: continue
        edges = [(walk[i],walk[i+1]) for i, _ in enumerate(walk[:-1])]
        for edge in edges:
            if edge not in count_dict: count_dict[edge] = 0
            count_dict[edge] += 1
    sum_ = sum(count_dict.values())
    count_dict = {k:np.round((v/sum_)*10,2) for k,v in count_dict.items()}
    count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))
    keys = list(count_dict.keys())[:topk]
    count_dict = {k:i+1 for i,k in enumerate(keys)}
    return count_dict

def visualize_rw_in_walk(file_name, model,hMM,hmm, extra_params=dict()):

    np.random.seed(42)
    edgecolor = '#c8cacc'
    node_size = 70   
    arrow_size = 6           # size of edge arrow (viz)
    edge_width = 0.5  
    scale_degree_size = 4000
    colors = {'min':'#ec8b67', 'maj':'#6aa8cb'}
    
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    main_directory = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/"
    graph_file = os.path.join(main_directory, file_name)
    g = read_graph(graph_file)
    walks = get_walks(g, model=model, extra_params=extra_params)
    walk_dict = count_edges(walks)
    from_min_edges = {(u,v):w for (u,v),w in walk_dict.items() if g.nodes[u]["m"] == 1}
    from_maj_edges = {(u,v):w for (u,v),w in walk_dict.items() if g.nodes[u]["m"] == 0}
    

    
    # pos = nx_pydot.graphviz_layout(g, prog='neato')
    pos = nx.spring_layout(g, k=0.5, iterations=20)
    node_color = [colors['min'] if obj['m'] else colors['maj'] for n,obj in g.nodes(data=True)]
    node2betn = nx.betweenness_centrality(g, normalized=True)
    node_dict = [node2betn[node] * scale_degree_size + 15 for node in g.nodes()]
    nx.draw_networkx_nodes(g, pos,  node_color=node_color, node_size=node_dict, ax=ax)
    edge_color = [edgecolor for e in g.edges()]
    # nx.draw_networkx_edges(g, pos, edgelist=g.edges(), edge_color=edgecolor, width=edge_width, arrows=True, arrowsize=arrow_size, ax=ax)
    
    edges = from_min_edges.keys()
    width =  list(from_min_edges.values())
    edge_colors = ["blue" for e in edges]
    nx.draw_networkx_edges(g, arrows=True, 
                           edgelist=edges, 
                           edge_color=edge_colors, pos=pos, ax=ax,alpha=0.9)
    
    edges = from_maj_edges.keys()
    width =  list(from_maj_edges.values())
    edge_colors = ["blue" for e in edges]
    nx.draw_networkx_edges(g, arrows=True,
                           edgelist=edges, 
                           edge_color=edge_colors, pos=pos, ax=ax,alpha=0.9)

    nx.draw_networkx_edge_labels(g, pos, edge_labels=walk_dict,label_pos=0.6)
    img_filename = "trial_{}_hMM{}_hmm{}.png".format(model,hMM,hmm)
    if "alpha" in extra_params.keys(): img_filename = "trial_{}_hMM{}_hmm{}_alpha{}.png".format(model,hMM,hmm,extra_params["alpha"])
    fig.savefig(img_filename, bbox_inches='tight')

def avg_indegree_due_to_grp(g, grp):
    node_attrs = nx.get_node_attributes(g, "group")
    itr = [node for node, _ in node_attrs.items() if get_label(_) == grp]
    total_len = len(itr)
    print(total_len, grp)
    sum_ = 0
    for i in itr:      
        neighbors = list(g.predecessors(i))
        diff_nghs = len([ngh for ngh in neighbors if get_label(node_attrs[ngh]) != grp])
        sum_ += diff_nghs
    avg_indg = sum_/total_len
    return avg_indg

def check_avg_indegree(fm=0.3):
    epsilon = 1e-6
    file_path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/DPAH_fm_{}".format(fm)
    graph_files = [os.path.join(file_path,file_name) for file_name in os.listdir(file_path) if "netmeta" not in file_name and ".gpickle" in file_name]
    for graph_file in graph_files:
        g = read_graph(graph_file)
        hMM, hmm = graph_file.split("hMM")[-1].split("-")[0], graph_file.split("hmm")[-1].split("-")[0]
        hMM, hmm = hMM.replace(".gpickle","").replace("_t_29",""), hmm.replace(".gpickle","").replace("_t_29","")
        avgmindegree, avgMindegree = avg_indegree_due_to_grp(g,"m"), avg_indegree_due_to_grp(g, "M")
        alpha_unno_m, alpha_unno_M = 1/(avgmindegree+epsilon), 1/(avgMindegree+epsilon)
        sum_pr = alpha_unno_m+alpha_unno_M
        print("hMM: {}, hmm:{}, avgmindegree: {}, avgMindegree: {}".format(hMM,hmm,avgmindegree, avgMindegree))
        pr_m, pr_M = alpha_unno_m/sum_pr, alpha_unno_M/sum_pr
        print("alpha pr m : {}, alpha pr M: {}".format(pr_m,pr_M)) 

def plot_utility_metrics(ds="rice",models=[]):
    # models = ["nonlocaladaptivealpha_beta_2.0", "indegree_beta_2.0",  "indegree_beta_-2.0", "indegree_beta_0.0","fw_p_1.0_q_1.0"]
    # models = ["nlindlocalind_alpha_0.0_beta_2.0","nlindlocalind_alpha_0.3_beta_2.0","nlindlocalind_alpha_0.5_beta_2.0","nlindlocalind_alpha_0.7_beta_2.0","nlindlocalind_alpha_1.0_beta_2.0", "fw_p_1.0_q_1.0"]
    labels = [label_dict.get(model, model) for model in models]
    x = np.arange(len(models))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots()
    seed_list =  [42,420,4200]
    # seed_list = [42]

    plot_dict = {"precision":[], "recall":[], "auc":[]}
    std_dict = {"precision":[], "recall":[], "auc":[]}
    for model in models:
        pre_list, recall_list, auc_list = [],[],[]
        for seed in seed_list:
            file_name = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/utility/model_{}_name_{}/seed_{}/_name{}.pkl".format(model,ds,seed,ds)
            pre, recall, acc, auc = print_utility(file_name)
            pre_list.append(pre)
            recall_list.append(recall)
            auc_list.append(auc)

        avg_pre = np.mean(pre_list)
        std_pre = np.std(pre_list)

        avg_recall = np.mean(recall_list)
        std_recall = np.std(recall_list)

        avg_auc = np.mean(auc_list)
        std_auc = np.std(auc_list)
        plot_dict["precision"].append(avg_pre)
        std_dict["precision"].append(std_pre)

        plot_dict["recall"].append(avg_recall)
        std_dict["recall"].append(std_recall)

        plot_dict["auc"].append(avg_auc)
        std_dict["auc"].append(std_auc)

    for label, val in plot_dict.items():
            if label != "auc": continue
            offset = width * multiplier
            rects = ax.bar(x + offset, val, width, label=label)
            ax.errorbar(x + offset, val,std_dict[label], fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
            ax.bar_label(rects, padding=7)
            multiplier += 1


    ax.set_ylabel('Utility Metrics')
    ax.set_xticks(x + width, labels)
    ax.legend(loc='upper right', ncols=3)
    ax.set_ylim(0, 1.1)
    fig.savefig("utility_barplot_{}.pdf".format(ds),bbox_inches='tight')

def plot_utility_metrics_syn(syn_ds,models,fm=0.3):
    labels = [label_dict.get(model, model) for model in models]
    x = np.arange(len(models))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout="constrained")
    # seed_list =  [42,420,4200]
    seed_list =  [42]

    plot_dict = {"precision":[], "recall":[], "auc":[]}
    std_dict = {"precision":[], "recall":[], "auc":[]}
    # plot_dict = {"auc":[]}
    # std_dict = {"auc":[]}
    for config in syn_ds:
        hMM, hmm = config.split(",")
        for model in models:
            pre_list, recall_list, auc_list = [],[],[]
            for seed in seed_list:
                file_name = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/utility/model_{}_fm_{}/seed_{}/_hMM{}_hmm{}.pkl".format(model,fm,seed,hMM,hmm)
                pre, recall, acc, auc = print_utility(file_name)
                pre_list.append(pre)
                recall_list.append(recall)
                auc_list.append(auc)
            # avg_pre /= len(seed_list)
            # avg_recall /= len(seed_list)
            # avg_acc /= len(seed_list)

            avg_pre = np.mean(pre_list)
            std_pre = np.std(pre_list)

            avg_recall = np.mean(recall_list)
            std_recall = np.std(recall_list)

            avg_auc = np.mean(auc_list)
            std_auc = np.std(auc_list)
            plot_dict["precision"].append(avg_pre)
            std_dict["precision"].append(std_pre)

            plot_dict["recall"].append(avg_recall)
            std_dict["recall"].append(std_recall)

            plot_dict["auc"].append(avg_auc)
            std_dict["auc"].append(std_auc)

    for label, val in plot_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, val, width, label=label)
            ax.errorbar(x + offset, val,std_dict[label], fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
            ax.bar_label(rects, padding=3)
            multiplier += 1


    ax.set_ylabel('Utility Metrics')
    ax.set_xticks(x + width,labels)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1.1)
    fig.savefig("utility_barplot_{}_fm{}.png".format(syn_ds[0],fm),bbox_inches='tight')


def plot_fair_metrics(ds="rice",models=[],t=29):
    labels = [label_dict.get(model, model) for model in models]
    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots()
    seed_list = [42,420,4200]

    plot_dict = dict()
    std_dict = dict()
    
    for model in models:
        vis_list = []
        for seed in seed_list:
            file_name = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t)
            vis_dict = print_visibility(file_name)
            print(vis_dict)
            vis_list.append(vis_dict)
  
        keys = vis_list[0].keys()
        for key in keys:
            arr = [_[key] for _ in vis_list]
            avg_vis, std_vis = np.mean(arr), np.std(arr)
            if key not in plot_dict: 
                plot_dict[key] = list()
                std_dict[key] = list()
            plot_dict[key].append(avg_vis)
            std_dict[key].append(std_vis)
  
    for label, val in plot_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, val, width, label=label)
            ax.errorbar(x + offset, val,std_dict[label], fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
            # ax.bar_label(rects, padding=3)
            multiplier += 1


    ax.set_ylabel('Visibility')
    ax.set_xticks(x + width, labels)
    ax.legend(loc='upper left', ncols=3)
    if ds == "rice": llim, uplim = 0,0.004
    elif ds == "twitter_climate": llim, uplim = 0,0.005
    else: llim, uplim = 0, 0.0025
    ax.set_ylim(llim,uplim)
    fig.savefig("vis_barplot_{}_{}.pdf".format(ds,t),bbox_inches='tight')

def plot_fair_metrics_syn(syn_ds,models,fm=0.3):
    config = syn_ds[0]
    labels = [label_dict.get(model, model) for model in models]
    x = np.arange(len(models))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots()
    # seed_list =  [42,420,4200]
    seed_list =  [42]

    plot_dict = dict() 
    std_dict = dict()
    
    hMM, hmm = config.split(",")
    
    for model in models:
        vis_list = []
        for seed in seed_list:
            # file_name = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/model_{}_fm_{}/seed_{}/{}-N1000-fm{}-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_29.gpickle".format(model,fm,seed,model,fm,hMM,hmm)
            file_name = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}_fm_{}/{}-N1000-fm{}-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_29.gpickle".format(model,fm,model,fm,hMM,hmm)
            vis_dict = print_visibility(file_name)
            vis_list.append(vis_dict)

   
        keys = vis_list[0].keys()
        for key in keys:
            arr = [_[key] for _ in vis_list]
            avg_vis = np.mean(arr)
            std_vis = np.std(arr)
            if key not in plot_dict: 
                plot_dict[key] = list()
                std_dict[key] = list()
            plot_dict[key].append(avg_vis)
            std_dict[key].append(std_vis)
    print(plot_dict, std_dict)
    for label, val in plot_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, val, width, label=label)
            ax.errorbar(x + offset, val,std_dict[label], fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
            multiplier += 1


    ax.set_ylabel('Visibility')
    ax.set_xticks(x + width,labels)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 0.004)
    fig.savefig("vis_barplot_{}_fm{}.png".format(config,fm),bbox_inches='tight')


def plot_fair_metrics_v2(ds="rice",models=[],t=29):
    labels = [label_dict.get(model, model) for model in models]
    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots()
    seed_list =  [42,420,4200]
    # seed_list = [42]

    plot_dict = dict()
    std_dict = dict()
    for model in models:
        fair_list = []
        for seed in seed_list:
            file_name = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t)
            fair_dict = print_fairness(file_name)
            fair_list.append(fair_dict)
            fair_list.append(fair_dict)

        # print(vis_list)   
        keys = fair_list[0].keys()
        for key in keys:
            arr = [_[key] for _ in fair_list]
            avg_fair = np.mean(arr)
            std_fair = np.std(arr)
            if key not in plot_dict: 
                plot_dict[key] = list()
                std_dict[key] = list()
            plot_dict[key].append(avg_fair)
            std_dict[key].append(std_fair)
  
    for label, val in plot_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, val, width, label=label)
            ax.errorbar(x + offset, val,std_dict[label], fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
            # ax.bar_label(rects, padding=3)
            multiplier += 1


    ax.set_ylabel('Fair Betweenness Centrality')
    ax.set_xticks(x + width, labels)
    ax.legend(loc='upper left', ncols=3)
    if ds == "rice": llim, uplim = -0.004,0.004
    elif ds == "twitter_climate": llim, uplim = -0.005,0.005
    else: llim, uplim = -0.0025, 0.0025
    ax.set_ylim(llim,uplim)
    fig.savefig("fair_barplot_{}_{}.pdf".format(ds,t),bbox_inches='tight')


def avg_indegree_due_to_itself(g, grp):
        node_attrs = nx.get_node_attributes(g,"group")
        itr = [node for node, _ in node_attrs.items() if _ == grp]
        total_len = len(itr)
        sum_ = 0
        total_sum_ = 0
        for i in itr:      
            neighbors = list(g.predecessors(i))
            diff_nghs = len([ngh for ngh in neighbors if node_attrs[ngh] == grp])
            sum_ += diff_nghs
            total_sum_ += len(neighbors)
        if total_sum_ == 0: return sum_
        avg_indg = sum_/total_sum_
        return avg_indg

def avg_outdegree_due_to_itself(g, grp):
        node_attrs = nx.get_node_attributes(g,"group")
        itr = [node for node, _ in node_attrs.items() if _ == grp]
        total_len = len(itr)
        sum_ = 0
        total_sum_ = 0
        for i in itr:      
            neighbors = list(g.successors(i))
            diff_nghs = len([ngh for ngh in neighbors if node_attrs[ngh] == grp])
            sum_ += diff_nghs
            total_sum_ += len(neighbors)
        if total_sum_ == 0: return sum_
        avg_outdg = sum_/total_sum_
        return avg_outdg

def avg_outdegree_to_grp_dict(g, grp):
        node_attrs = nx.get_node_attributes(g,"group")
        itr = [node for node, _ in node_attrs.items() if _ == grp]
        total_len = len(itr)
        sum_ = 0
        total_sum_ = 0
        out_dict = dict()

        for i in itr:      
            neighbors = list(g.successors(i))
            diff_nghs = [ngh for ngh in neighbors if node_attrs[ngh] != grp]
            
            for diff_ngh in diff_nghs:
                id_ = node_attrs[diff_ngh]
                if id_ not in out_dict: out_dict[id_] = 0
                out_dict[id_] += 1 
            
            total_sum_ += len(neighbors)
        
        avg_outdg = {k:v if v==0 else v/total_sum_ for k,v in out_dict.items()}
        return avg_outdg

def avg_indegree_to_grp_dict(g, grp):
        node_attrs = nx.get_node_attributes(g,"group")
        itr = [node for node, _ in node_attrs.items() if _ == grp]
        total_len = len(itr)
        sum_ = 0
        total_sum_ = 0
        in_dict = dict()

        for i in itr:      
            neighbors = list(g.predecessors(i))
            diff_nghs = [ngh for ngh in neighbors if node_attrs[ngh] != grp]
            
            for diff_ngh in diff_nghs:
                id_ = node_attrs[diff_ngh]
                if id_ not in in_dict: in_dict[id_] = 0
                in_dict[id_] += 1 
            
            total_sum_ += len(neighbors)
        
        avg_indg = {k:v if v==0 else v/total_sum_ for k,v in in_dict.items()}
        return avg_indg

def get_heg(g):
    hete_dict = dict()
    node_attrs = nx.get_node_attributes(g,"group")
    uniquegroups = list(set(node_attrs.values()))
    for group in uniquegroups:
        i_dict = avg_indegree_to_grp_dict(g,group)
        o_dict = avg_indegree_to_grp_dict(g,group)
        i_gg = avg_indegree_due_to_itself(g,group)
        o_gg = avg_outdegree_due_to_itself(g,group)

        he_g = 0     
        for k, v in i_dict.items():
            g_bar = k
            q_in_gbar_to_g = v
            q_out_g_to_gbar = o_dict[g_bar]
            he_t = q_in_gbar_to_g * o_gg
            he_b = i_gg * q_out_g_to_gbar
            he_g_gbar = (he_t+he_b)
            he_g += he_g_gbar

        hete_dict[group] = he_g/len(i_dict)
    
    return hete_dict

def get_hog(g):
    homo_dict = dict()
    node_attrs = nx.get_node_attributes(g,"group")
    uniquegroups = list(set(node_attrs.values()))
    for group in uniquegroups:
        i_gg = avg_indegree_due_to_itself(g,group)
        o_gg = avg_outdegree_due_to_itself(g,group)
        h_gg = i_gg*o_gg
        homo_dict[group] = h_gg
    return homo_dict

def plot_heg_hog(hMM,hmm):
    fig, ax = plt.subplots(nrows=1, ncols=1) 
    cmap = "coolwarm"
    norm = plt.Normalize(-0.004, 0.004)

    models = ["fw_p_1.0_q_1.0","indegreevarybetav2_beta_2.0"] 
    colors = {0:"#ffa700",1:"#0057e7"}
    linestyle = {models[0]:"dashed",models[1]:"solid"}
    marker = {0:"^", 1:"o"}
    group_dict = {0:"M", 1:"m"}

    for model in models:
        path =  "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_29.gpickle".format(model,model,hMM,hmm)
        fair_dict = print_fairness(path)
        g = read_graph(path)
        node_attrs = nx.get_node_attributes(g,"group")
        uniquegroups = list(set(node_attrs.values()))
        homo_dict = get_hog(g)
        hete_dict = get_heg(g)
        xs = [homo_dict[g] for g in uniquegroups]
        ys = [hete_dict[g] for g in uniquegroups]
        c = [colors[g] for g in uniquegroups]
        marker =  [marker[g] for g in uniquegroups]
        print("~~ model : {}, homo dict: {}".format(model,homo_dict))
        print("~~ model : {}, hete dict: {}".format(model,hete_dict))

        ax.plot(xs,ys,label=label_dict[model],color="#abb4bb",linestyle=linestyle[model])
        for g in uniquegroups:
            ax.scatter(homo_dict[g],hete_dict[g],marker=marker[g],c=fair_dict[g], cmap=cmap,norm=norm,label=group_dict[g])

    ax.set_xlabel("Homophilic Activity of Group g "+r"$h_o(g)$")
    ax.set_ylabel("Heterophilic Activity of Group g "+r"$h_e(g)$")
    ax.set_ylim(-0.05,1)
    ax.set_xlim(0,1)
    ax.legend(loc = "upper right",bbox_to_anchor=(1.0,1.0))
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(smap, ax=ax, fraction=0.1, shrink = 0.8, orientation="horizontal",ticks=[-0.004, 0, 0.004]) 
 
    fig.savefig('plots/heghog_hMM{}_hmm{}.pdf'.format(hMM,hmm),bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window
        

def draw_graph(file_name,hMM=0.0,hmm=0.0):

    np.random.seed(42)
    edgecolor = '#c8cacc'
    node_size = 70   
    arrow_size = 6           # size of edge arrow (viz)
    edge_width = 0.5  
    scale_degree_size = 4000
    colors = {'min':'#ec8b67', 'maj':'#6aa8cb'}
    
    fig, ax = plt.subplots(1,1)
    main_directory = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/"
    graph_file = os.path.join(main_directory, file_name)
    graph_file = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/DPAH_fm_0.3/DPAH-N55-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}-ID0.gpickle".format(hMM,hmm)
    g = read_graph(graph_file)
    node_attr = nx.get_node_attributes(g, "group")

    
    # pos = nx_pydot.graphviz_layout(g, prog='neato')
    pos = nx.spring_layout(g, k=0.5, iterations=20)
    for node, p in pos.items():
        if node_attr[node] == 0:
            pos[node] = [p[0]-1,p[1]]
        else:
            pos[node] = [p[0]+1,p[1]]
    node_color = [colors['min'] if obj['m'] else colors['maj'] for n,obj in g.nodes(data=True)]
    node2betn = nx.betweenness_centrality(g, normalized=True)
    node_dict = [node2betn[node] * scale_degree_size + 15 for node in g.nodes()]
    nx.draw_networkx_nodes(g, pos,  node_color=node_color, node_size=node_dict, ax=ax)

    cross_edges = [(u,v) for u,v in g.edges() if node_attr[u] != node_attr[v]]
    non_cross_edges = list(set(g.edges()) - set(cross_edges))
    edge_color = [edgecolor for e in g.edges()]
    nx.draw_networkx_edges(g, pos, edgelist=cross_edges, edge_color=edgecolor, width=edge_width, arrows=True, arrowsize=arrow_size, ax=ax)
    nx.draw_networkx_edges(g, pos, edgelist=non_cross_edges, edge_color=edgecolor, width=edge_width, arrows=True, arrowsize=arrow_size, ax=ax)
    

    img_filename = "trial_hMM{}_hmm{}.pdf".format(hMM,hmm)
    fig.savefig(img_filename, bbox_inches='tight')           

if __name__ == "__main__":
    ds = "facebook_locale"

    models = ["n2v_p_1.0_q_1.0","fw_p_1.0_q_1.0","indegreevarybetav2_beta_2.0","nlindlocalind_alpha_1.0_beta_2.0"]
    t = 29
    plot_fair_metrics(ds=ds,models=models,t=t)
    plot_fair_metrics_v2(ds=ds,models=models,t=t)
    plot_utility_metrics(ds=ds,models=models)
    # plot_heg_hog(hMM=0.1,hmm=1.0)
    # plot_heg_hog(hMM=0.8,hmm=0.3)
    # plot_heg_hog(hMM=0.6,hmm=1.0)

        


