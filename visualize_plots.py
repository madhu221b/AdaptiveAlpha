import os
import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import operator

from org.gesis.lib.io import create_subfolders
from org.gesis.lib.metrics_utils import get_statistical_imparity, get_er_nwlevel, get_er_userlevel,  get_disparity

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

label_dict = {
 "nlindlocalind_alpha_1.0_beta_2.0": "NonLocal Walker", 
"n2v_p_1.0_q_1.0": "n2v",
"baseline": "Baseline",
# "ffw_p_1.0_q_1.0": "Fairwalk",
# "fcw_p_2_alpha_0.5": "Crosswalk",
"ffw_p_1.0_q_1.0": "Fairwalk",
"fcw_p_2_alpha_0.5": "Crosswalk",
"fpr_psi_0.3": "Fair PageRank",
"random": "Random",
"fastadaptivealphatest_beta_2.0":"Varying "+r"$\alpha$",
"fastadaptivealphatestfixed_alpha_0.3_beta_2.0": r"$\alpha$"+"=0.3",
"fastadaptivealphatestfixed_alpha_0.5_beta_2.0": r"$\alpha$"+"=0.5",
"fastadaptivealphatestfixed_alpha_0.7_beta_2.0": r"$\alpha$"+"=0.7",
"fastadaptivealphatestfixed_alpha_1.0_beta_2.0": r"$\alpha$"+"=1.0",
}

# if os.environ.get('DISPLAY','') == '':
#     print('no display found. Using non-interactive Agg backend')
#     os.environ.__setitem__('DISPLAY', ':0.0')
#     mpl.use('Agg')

os.environ['MPLCONFIGDIR'] = os.path.join(os.getenv('HOME'), '.config', 'matplotlib')
# import matplotlib.pyplot as plt


from org.gesis.lib.n2v_utils import get_avg_group_centrality, read_graph, get_centrality_dict, get_diff_group_centrality
from generate_heatmap_centrality import get_grid

T = 30
hMM_list, hmm_list = np.arange(0,1.1,0.1), np.arange(0,1.1,0.1)
MAIN_SEEDS = [42,420,4200]
main_path = "../AdaptiveAlpha/"

def print_visibility(file_name):
 
    if not os.path.exists(file_name) and "baseline" not in file_name: 
        print(file_name)
        return dict()
    seed = file_name.split("seed_")[-1].split("/")[0]
    g = read_graph(file_name,seed=seed)
    vis_dict = dict() 
    groups = list(set(nx.get_node_attributes(g, "group").values()))
    print("groups:", groups)
    
    # elif "twitter" in file_name: min_grp, maj_grp = 2, 1
    cent_file = file_name.replace(".gpickle","") + ".pkl"
    if not os.path.exists(cent_file):
        create_subfolders(cent_file)
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
 
    if not os.path.exists(file_name) and "baseline" not in file_name: 
        print(file_name)
        return dict()
    seed = file_name.split("seed_")[-1].split("/")[0]
    g = read_graph(file_name,seed=seed)
    fairness_dict = dict() 
    groups = list(set(nx.get_node_attributes(g, "group").values()))
    

    cent_file = file_name.replace(".gpickle","") + ".pkl"
    if not os.path.exists(cent_file):  cent_file = file_name.replace(".gpickle","") + ".pickle"
    if not os.path.exists(cent_file):
         create_subfolders(cent_file)
         centrality_dict = nx.betweenness_centrality(g, normalized=True)
         print("Generating pkl file: ", cent_file)
         with open(cent_file, 'wb') as f:                
                pkl.dump(centrality_dict,f)
    else:
        print("Loading cent file: ", cent_file)
        with open(cent_file,"rb") as f:
              centrality_dict = pkl.load(f)
    
    if "tuenti" in file_name: group_list = [2]
    elif "pokec" in file_name: group_list = [0]
    else:  group_list = [1]
    for group in group_list:
        avg_diff =  get_diff_group_centrality(g,centrality_dict,group=group)
        fairness_dict[group] = avg_diff
    return fairness_dict

def print_utility(file_name, t=29):
    prec, recall, acc, auc = 0, 0, 0, 0
    try:
  
        if os.path.exists(file_name):
            with open(file_name,"rb") as f:
                result_dict = pkl.load(f)
            prec, recall, acc = result_dict[t].get("precision",0), result_dict[t].get("recall",0), result_dict[t].get("accuracy",0)
            auc = result_dict[t].get("auc_score",0)
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
    file_path = main_path+"DPAH_fm_{}".format(fm)
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

def plot_utility_metrics(ds="rice",models=[],t=29):
    # models = ["nonlocaladaptivealpha_beta_2.0", "indegree_beta_2.0",  "indegree_beta_-2.0", "indegree_beta_0.0","fw_p_1.0_q_1.0"]
    # models = ["nlindlocalind_alpha_0.0_beta_2.0","nlindlocalind_alpha_0.3_beta_2.0","nlindlocalind_alpha_0.5_beta_2.0","nlindlocalind_alpha_0.7_beta_2.0","nlindlocalind_alpha_1.0_beta_2.0", "fw_p_1.0_q_1.0"]
    labels = [label_dict.get(model, model) for model in models]
    x = np.arange(len(models))  # the label locations
    width = 0.15 # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots()
    seed_list =  [42,420,4200]
    # seed_list =  [42]

    plot_dict = {"precision":[], "recall":[], "auc":[]}
    std_dict = {"precision":[], "recall":[], "auc":[]}
    for model in models:
        pre_list, recall_list, auc_list = [],[],[]
        for seed in seed_list:
            file_name = main_path+"utility/model_{}_name_{}/seed_{}/_name{}.pkl".format(model,ds,seed,ds)
            pre, recall, acc, auc = print_utility(file_name, t=t)
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
            ax.bar_label(rects, padding=7,fmt='%.2f', fontsize=12)
            multiplier += 1


    # ax.set_ylabel('Utility Metrics')
    ax.set_xticks(x + width, labels)
    ax.set_xticklabels(labels, fontsize=16, rotation=45, ha='right')
    ax.legend(loc='upper right', ncols=3)
    ax.set_ylim(0, 1.1)
  
    ax.set_xticklabels(labels)
    # ax.yaxis.get_label().set_fontsize(12)
    ax.tick_params(axis='y', labelsize=14)
    fig.savefig("utility_barplot_{}_t_{}.pdf".format(ds,t),bbox_inches='tight')

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
                file_name = main_path+"utility/model_{}_fm_{}/seed_{}/_hMM{}_hmm{}.pkl".format(model,fm,seed,hMM,hmm)
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
    fig, ax = plt.subplots(figsize=(10,5))
    seed_list = [42,420,4200]

    plot_dict = dict()
    std_dict = dict()
    
    for model in models:
        vis_list = []
        for seed in seed_list:
            file_name = main_path+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t)
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
    else: llim, uplim = -0.006, 0.006
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
            file_name = main_path+"{}_fm_{}/{}-N1000-fm{}-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_29.gpickle".format(model,fm,model,fm,hMM,hmm)
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
    fig.savefig("../AdaptiveAlpha/vis_barplot_{}_fm{}.pdf".format(config,fm),bbox_inches='tight')
    # plt.show()

def plot_fair_metrics_v2(ds="rice",models=[],t=29):
    labels = [label_dict.get(model, model) for model in models]
    x = np.arange(len(models))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(figsize=(10,5))
    seed_list =  [42,420,4200]
    # seed_list =  [42]

    plot_dict = dict()
    std_dict = dict()
    for model in models:
        fair_list = []
        for seed in seed_list:
            file_name = main_path+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t)
            fair_dict = print_fairness(file_name)
            print(fair_dict)
            if ds == "pokec":
               fair_dict = {1:fair_dict[0]}
            if ds == "rice":
                fair_dict = {2:fair_dict[1]}
            fair_list.append(fair_dict)
            # fair_list.append(fair_dict)
            # print(fair_dict)

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
    
    print(plot_dict, std_dict)
    for label, val in plot_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, val, width, color="orange",label=label)
            ax.errorbar(x + offset, val,std_dict[label], fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
            # ax.bar_label(rects, padding=3)
            multiplier += 1


    # ax.set_ylabel('Betweenness Centrality Disparity')
    ax.set_xticks(x + width, labels)
    ax.set_xticklabels(labels, fontsize=18, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=15)
    ax.legend(loc='upper left', ncols=3)
    # if ds == "rice": llim, uplim = -0.004,0.004
    if ds in "rice": 
        llim, uplim = 0, 0.004
        ax.set_ylim(llim,uplim)
    elif ds == "twitter": 
         llim, uplim = -0.00007,0.00007
         ax.set_ylim(llim,uplim)
    else: 
        pass
    # elif ds == "tuenti": llim, uplim = -10e-9,10e-9
    # else: llim, uplim =  -0.008, 0.008
  
 
    ax.axhline(y=0, color="black",linewidth=.7)
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
        path =  main_path+"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_29.gpickle".format(model,model,hMM,hmm)
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
        

def draw_graph(hMM=0.0,hmm=0.0):

    np.random.seed(42)
    edgecolor = '#c8cacc'
    node_size = 70   
    arrow_size = 6           # size of edge arrow (viz)
    edge_width = 0.5  
    scale_degree_size = 4000
    colors = {'min':'#ec8b67', 'maj':'#6aa8cb'}
    
    fig, ax = plt.subplots(1,1)
    graph_file = main_path+"DPAH_fm_0.3/DPAH-N30-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}-ID0.gpickle".format(hMM,hmm)
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


def plot_kde(filename):
    """
        
    # sns.kdeplot(data=betn_dfM["betweenness"],color="blue",fill=True,ax=ax)
    # sns.kdeplot(data=betn_dfm["betweenness"].cumsum(),color="orange",fill=True,ax=ax)

    """
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))

    g = read_graph(filename)
    node_attr = nx.get_node_attributes(g, "group")
    betn = nx.betweenness_centrality(g,normalized=True)
    betnM = {k:v for k,v in betn.items() if node_attr[k] == 0}
    betnm = {k:v for k,v in betn.items() if node_attr[k] == 1}

    betn_df = pd.DataFrame.from_dict(betn, orient='index', columns=['betweenness'])
    betn_dfM = pd.DataFrame.from_dict(betnM, orient='index', columns=['betweenness'])
    betn_dfm = pd.DataFrame.from_dict(betnm, orient='index', columns=['betweenness'])
    
    # sns.distplot(betn_dfM, hist=True, label="betn", ax=ax1)
    # sns.distplot(betn_dfm, hist=True, label="betn", ax=ax1)
    sns.kdeplot(betn_dfM, ax=ax1)
    sns.kdeplot(betn_dfm, ax=ax1)
    ax1.set_title('basic distplot (kde=True)')
    lines = ax1.get_lines()
   
    # get distplot line points
    lineM, linem = ax1.get_lines()[0], ax1.get_lines()[1]
    xdM, xdm = lineM.get_xdata(), linem.get_xdata()
    ydM, ydm = lineM.get_ydata(), linem.get_ydata()
    # https://stackoverflow.com/questions/29661574/normalize-numpy-array-columns-in-python
    
    def normalize(x,x_all):
        return (x - x_all.min(0)) / x_all.ptp(0)
    #normalize points)
    y_all = np.hstack([ydM,ydm])
    ydM = normalize(ydM,y_all)
    ydm = normalize(ydm,y_all)
    # plot them in another graph
    ax2.plot(xdM, ydM, color="blue",label="Majority")
    ax2.plot(xdm, ydm, color="orange",label="Minority")
    # ax2.set_title('basic distplot (kde=True)\nwith normalized y plot values')
    ax2.legend()
    ax1.set_visible(False)


    ax2.set_xlabel("Betweenness Centrality")
    ax2.set_ylabel("Normalized PDF")
    # ax2.set_xlim(right=0.3)
    # sns_plot = sns.displot(data=list(betn_df["betweenness"]))
    fig.savefig("output.pdf")

def plot_kde_v2(filename):
    """
        
    # sns.kdeplot(data=betn_dfM["betweenness"],color="blue",fill=True,ax=ax)
    # sns.kdeplot(data=betn_dfm["betweenness"].cumsum(),color="orange",fill=True,ax=ax)

    """
    fig, ax = plt.subplots(1,1)

    g = read_graph(filename)
    node_attr = nx.get_node_attributes(g, "group")
    betn = nx.betweenness_centrality(g, normalized=True)
    betn_df = pd.DataFrame.from_dict(betn, orient='index', columns=['betweenness'])
    indegree_df = pd.DataFrame.from_dict(dict(g.in_degree()), orient='index', columns=['indegree'])
    indegree_df = indegree_df/indegree_df.sum()
    combined_df = pd.concat([betn_df,indegree_df],axis=1)
   
    sns.kdeplot(data=combined_df, x="betweenness", y="indegree",fill=True)

    ax.set_xlabel("Betweenness Centrality")
    ax.set_ylabel("Indegree Centrality")
    ax.set_xlim(right=0.01)
    ax.set_ylim(top=0.006)
    # sns_plot = sns.displot(data=list(betn_df["betweenness"]))
    fig.savefig("output.pdf")

def get_statistical_imparity_all(models, use_syn_ds=False, hMM=0.0, hmm=0.0, ds=""):

    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    
    print("Dataset: ", ds)
    mean_list, std_list = list(), list()
    for model in models:
        model_sp = list()
        for seed in MAIN_SEEDS:
            val = get_statistical_imparity(model,use_syn_ds=use_syn_ds,hMM=hMM,hmm=hmm,ds=ds,seed=seed)
            model_sp.append(val)
        mean_sp, var_sp = np.mean(model_sp), np.var(model_sp)
        mean_list.append(mean_sp)
        std_list.append(var_sp)
        print("Mean SP : {:2e} , Var SP : {:2e} for Model:  {}".format(mean_sp, var_sp, model))
    
    ax.errorbar([label_dict[_] for _ in models], mean_list, label=ds, marker="o", yerr=std_list)


    ax.set_xlabel("Methods")
    ax.set_ylabel("Statistical Parity")
    ax.invert_yaxis()
    ax.legend(loc = "lower right",bbox_to_anchor=(0.4,0.0))
    fig.savefig("statistical_imparity.pdf",bbox_inches='tight')   # save the figure to file
    plt.close(fig)



def get_er_network_all(models, use_syn_ds=False, hMM=0.0, hmm=0.0, ds=""):
    """
    Scaled down by 10**4
    """
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    allvals = list()
    for ds in ["rice","facebook"]:
        print("DS: ", ds)
        mean_list, std_list = list(), list()
        for model in models:
            model_er = list()
            for seed in MAIN_SEEDS:
                val = get_er_nwlevel(model,use_syn_ds=use_syn_ds,hMM=hMM,hmm=hmm,ds=ds,seed=seed)
                model_er.append(val)
            model_er = np.array(model_er)/10**4
    
            mean_er, var_er = np.mean(model_er), np.var(model_er)
            print("Mean ER : {:2e} , Var ER : {:2e} for Model:  {}".format(mean_er, var_er, model))
            mean_list.append(mean_er)
            allvals.extend(mean_list)
            std_list.append(var_er)
        
        ax.errorbar([label_dict[_] for _ in models], mean_list, label=ds, marker="o", yerr=std_list)


    ax.set_xlabel("Methods")
    ax.set_ylabel("Equality of Representation at Network Level")
    # ax.set_yscale("log")
    # ax.set_ylim(np.min(allvals), np.max(allvals))
    ax.invert_yaxis()
    ax.legend(loc = "lower right",bbox_to_anchor=(0.4,0.0))
    fig.savefig("er_nw.pdf",bbox_inches='tight')   # save the figure to file
    plt.close(fig)

def get_disparity_all(models, use_syn_ds=False, hMM=0.0, hmm=0.0, ds=""):

    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    for ds in ["rice", "facebook"]:
        print("Dataset: ", ds)
        mean_list, std_list = list(), list()
        for model in models:
            model_dis = list()
            for seed in MAIN_SEEDS:
                val = get_disparity(model,use_syn_ds=use_syn_ds,hMM=hMM,hmm=hmm,ds=ds,seed=seed)
                model_dis.append(val)
            mean_dis, var_dis = np.mean(model_dis), np.var(model_dis)
            mean_list.append(mean_dis)
            std_list.append(var_dis)
            print("Mean DIS : {:2e} , Var DIS : {:2e} for Model:  {}".format(mean_dis, var_dis, model))
        
        ax.errorbar([label_dict[_] for _ in models], mean_list, label=ds, marker="o", yerr=std_list)


    ax.set_xlabel("Methods")
    ax.set_ylabel("Disparity")
    ax.invert_yaxis()
    ax.legend(loc = "lower right",bbox_to_anchor=(0.4,0.0))
    fig.savefig("disparity.pdf",bbox_inches='tight')   # save the figure to file
    plt.close(fig)

def plot_betn_vs_indegree(model,hMM,hmm):
    fig, ax = plt.subplots(nrows=1, ncols=1) 


    # path =  main_path+"DPAH_fm_0.3/DPAH-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}-ID0.gpickle".format(hMM,hmm)
    path = os.path.join(main_path,"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,29))
    g = read_graph(path)
    bet = nx.betweenness_centrality(g, normalized=True)
    ind = g.in_degree()
    bet_list , ind_list = list(), list()
    for k, v in bet.items():
        bet_list.append(v)
        ind_list.append(ind[k])
    ax.scatter(bet_list,ind_list)

    ax.set_xlabel("Betweenness Centrality")
    ax.set_ylabel("Indegree Centrality")

    fig.savefig('scatter.pdf'.format(hMM,hmm),bbox_inches='tight')   # save the figure to file
    plt.close(fig)   

def get_line_plot_t_vs_utility(ds, models):
    fig, ax = plt.subplots(nrows=1, ncols=1) 
    

    for model in models:
        xs,ys,y_error = list(), list(), list()

        for i, t in enumerate(range(T)):
            auc_list = []
            for seed in MAIN_SEEDS:
                file_name = main_path+"utility/model_{}_name_{}/seed_{}/_name{}.pkl".format(model,ds,seed,ds)
                _,_,_, auc = print_utility(file_name,t=t)
                auc_list.append(auc)
            
            mean_auc, std_auc = np.mean(auc_list), np.std(auc_list)
            xs.append(i)
            ys.append(mean_auc)
            y_error.append(std_auc)
            
        ax.errorbar(xs,ys, label=label_dict[model], marker="o", yerr = y_error)
    

    
    # idxs = [_ for _ in range(len(dict_))]
    # ax.set_xticks(idxs)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.0)
    ax.legend(loc = "lower right",bbox_to_anchor=(0.7,0))
    fig.savefig('plots/time_vs_utility_ds_{}.pdf'.format(ds),bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window

def get_new_recos_ds(ds, models):
    markerstyle = {models[0]:"o",models[1]:"^"}
    colors = {"0->0":"orange",
              "0->1": "b",
              "1->1": "r",
              "1->0":"g"}

    fig, ax = plt.subplots(nrows=1, ncols=1) 
    for model in models:
        xs,ys,y_error = list(), dict(), dict()
        for i, time in enumerate(range(T)):

            reco_dict = dict()
            for seed in MAIN_SEEDS:
                if time == 0:
                    prev_gpath = main_path+"data/{}/{}_{}.gpickle".format(ds,ds,seed)
                else: 
                    prev_gpath = os.path.join(main_path,"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,time-1))

                curr_gpath = os.path.join(main_path,"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,time))
            
                g_prev = read_graph(prev_gpath)
                g_curr = read_graph(curr_gpath)
            
                node_attrs = nx.get_node_attributes(g_curr,"group")
                recos = list(set(g_curr.edges())-set(g_prev.edges())) # get the total recommendations
                assert len(recos) == g_curr.number_of_nodes()
                # count the number of recos by identity
                count_dict = dict()
                for (u,v) in recos:
                    label = "{}->{}".format(node_attrs[u],node_attrs[v])
                    if label not in count_dict: count_dict[label] = 0
                    count_dict[label] += 1

                sum_ = sum(count_dict.values())
                for k,v in count_dict.items():
                    if k not in reco_dict: reco_dict[k] = list()
                    reco_dict[k].append((v/sum_))

            # aggregate
            xs.append(i)
            for k,v in reco_dict.items():
                if k not in ys: ys[k] = list()
                if k not in y_error: y_error[k] = list()
                ys[k].append(np.mean(v))
                y_error[k].append(np.std(v))


        for i, (k, v) in enumerate(ys.items()):
           ax.errorbar(xs,v, label=k, marker=markerstyle[model], yerr = y_error[k],color=colors[k])    
    
    dummy_lines = []
    for linestyle in ["-","--"]:
        dummy_lines.append(ax1.plot([],[], c="black")[0])

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Percent of new recommendations")
    ax.legend(loc = "lower right",bbox_to_anchor=(1,0.7))
    ax.set_ylim(0, 1.0)
    fig.savefig('plots/time_vs_newperc_ds_{}.pdf'.format(ds),bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window  
                
               

def get_avg_metric_all(models, ds, metric_="betweenness", t=29):
    if ds == "tuenti": groups = [1,2]
    elif ds == "rice": groups = [0,1]
    fig, ax = plt.subplots()

    data = dict()
    data_std = dict()
    # populate empty dict
    data[metric_+"_avg"] = list()
    data_std[metric_+"_avg"] = list()
    for group in groups:
        data[metric_+"_"+str(group)] = list()
        data_std[metric_+"_"+str(group)] = list()
    
    for model in models:
        avg_val = list()
        avg_group_val = list()
        for seed in MAIN_SEEDS:
            if metric_ == "betweenness":
                cent_file = main_path+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.pickle".format(model,ds,seed,model,ds,t)
                if not os.path.exists(cent_file): cent_file = main_path+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.pkl".format(model,ds,seed,model,ds,t)
                print("Loading cent file: ", cent_file)
                with open(cent_file,"rb") as f:
                            centrality_dict = pkl.load(f)
                            mean_val = np.mean(list(centrality_dict.values())) 
                            avg_val.append(mean_val)

                            g_file = main_path+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t)
                            g = read_graph(g_file, seed=seed)
                            avg_betns_by_group = [get_avg_group_centrality(g,centrality_dict,group=group) for group in groups]
                            avg_group_val.append(avg_betns_by_group)
            elif metric_ == "pagerank":
                cent_file = main_path+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}_pr.pkl".format(model,ds,seed,model,ds,t)
                print("Loading cent file: ", cent_file)
                with open(cent_file,"rb") as f:
                            centrality_dict = pkl.load(f)
                            mean_val = np.mean(list(centrality_dict.values())) 
                            avg_val.append(mean_val)

                            g_file = main_path+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t)
                            g = read_graph(g_file, seed=seed)
                            avg_prs_by_group = [get_avg_group_centrality(g,centrality_dict,group=group) for group in groups]
                            avg_group_val.append(avg_prs_by_group)
            elif metric_ == "clustering":
                cent_file = main_path+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}_clustering.pkl".format(model,ds,seed,model,ds,t)
                print("Loading cent file: ", cent_file)
                with open(cent_file,"rb") as f:
                            centrality_dict = pkl.load(f)
                            mean_val = np.mean(list(centrality_dict.values())) 
                            avg_val.append(mean_val)

                            g_file = main_path+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t)
                            g = read_graph(g_file, seed=seed)
                            avg_prs_by_group = [get_avg_group_centrality(g,centrality_dict,group=group) for group in groups]
                            avg_group_val.append(avg_prs_by_group)
            
            elif metric_ == "indegree":
                g_file = main_path+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t)
                g = read_graph(g_file, seed=seed)
                indegree = dict(g.in_degree())
                mean_val = np.mean(list(indegree.values())) 
                avg_val.append(mean_val)
                node_attrs = nx.get_node_attributes(g, "group")
                avg_ind_by_group = [np.mean([_ for k,_ in indegree.items() if node_attrs[k] == group]) for group in groups]
                avg_group_val.append(avg_ind_by_group)
            elif metric_ == "outdegree":
                g_file = main_path+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t)
                g = read_graph(g_file, seed=seed)
                indegree = dict(g.out_degree())
                mean_val = np.mean(list(indegree.values())) 
                avg_val.append(mean_val)
                node_attrs = nx.get_node_attributes(g, "group")
                avg_ind_by_group = [np.mean([_ for k,_ in indegree.items() if node_attrs[k] == group]) for group in groups]
                avg_group_val.append(avg_ind_by_group)




        data[metric_+"_avg"].append(np.mean(avg_val))       
        data_std[metric_+"_avg"].append(np.std(avg_val))    
        for i, group in enumerate(groups):
           vals = [_[i] for _ in avg_group_val]
           data[metric_+"_"+str(group)].append(np.mean(vals))
           data_std[metric_+"_"+str(group)].append(np.std(vals))

       

    
    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0.

    fig, ax = plt.subplots(layout='constrained')

    for metric, vals in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset,vals, width, label=metric)
        ax.errorbar(x + offset,vals, data_std[metric],label="",fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize =2)
        # ax.bar_label(rects, padding=7)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.tight_layout()
    ax.set_ylabel(metric_)
    ax.set_title('Models')
    ax.set_xticks(x + width, [label_dict.get(model, model) for model in models])
    ax.legend(loc='upper left', ncols=1, bbox_to_anchor=(1, 1))
    fig.savefig("trial_{}_{}.pdf".format(metric_,ds),bbox_inches='tight')
   
def get_indegree_diff_by_group(g):
    node_attrs = nx.get_node_attributes(g, "group")
    groups = list(set(nx.get_node_attributes(g, "group").values())) 
    indegree = dict(g.in_degree())
    avg_indegree = {group:np.mean([_ for k,_ in indegree.items() if node_attrs[k] == group]) for group in groups}
    if 0 in avg_indegree: avg_indegree = {0: avg_indegree[0]-avg_indegree[1]} # rice = 1 - 0, pokec = 0: {0 -1} 
    elif 2 in avg_indegree: avg_indegree = {2: avg_indegree[2]-avg_indegree[1]} # tuenti = 2 -1
    return avg_indegree

def get_models_vs_indegree(ds="rice",models=[],t=29):
    
    labels = [label_dict.get(model, model) for model in models]
    labels = labels[1:]
    x = np.arange(len(models) - 1)  # the label locations

    width = 0.12  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots()
    seed_list =  [42,420,4200]
    # seed_list =  [42]
    color_dict = {1: "orange", 2: "tab:blue"}
    plot_dict = dict()
    std_dict = dict()
    for model in models:
        print("model : ", model)
        fair_list = []
        for seed in seed_list:
            file_name = main_path+"model_{}_name_{}/seed_{}/_{}-name_{}_t_{}.gpickle".format(model,ds,seed,model,ds,t)
            g = read_graph(file_name, seed=seed)
            fair_dict = get_indegree_diff_by_group(g)
            fair_list.append(fair_dict)


        keys = fair_list[0].keys()
        for key in keys:
            arr = [_[key] for _ in fair_list]
            avg_fair = np.mean(arr)
            std_fair = np.std(arr)
            if ds != "tuenti":
                if key == 0: key = 1
                elif key == 1: key = 2
            if key not in plot_dict: 
                plot_dict[key] = list()
                std_dict[key] = list()
            
            plot_dict[key].append(avg_fair)
            std_dict[key].append(std_fair)
    

    for label, val in plot_dict.items():
            val = val[1:]
            offset = width * multiplier
            rects = ax.bar(x + offset, val, width, label=label, color="orange")
            ax.errorbar(x + offset, val,std_dict[label][1:], fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
            # ax.bar_label(rects, padding=3)
            

    fig.tight_layout() 
    
    # ax.set_ylabel(r"${\overline{ind}(2)} -  {\overline{ind}(1)}$", fontsize=16)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xticks(x + width, labels)
    ax.set_xticklabels(labels, fontsize=18, rotation=45, ha='right')
    ax.legend(loc='upper right', ncols=1, bbox_to_anchor=(1, 1.05), borderaxespad=0.)


    # ax.yaxis.get_label().set_fontsize(12) # pokec == plot_dict[1][0]
    ax.axhline(y=plot_dict[1][0], color='grey', linestyle='--', linewidth=2, label=f'Baseline')
    fig.savefig("indegree_barplot_{}_{}.pdf".format(ds,t),bbox_inches='tight')
    
if __name__ == "__main__":
    # ds = "facebook_locale" "cw_p_4_alpha_0.7",
    ds = "rice"
    models = ["baseline","random","ffw_p_1.0_q_1.0","fcw_p_2_alpha_0.5","fpr_psi_0.3","fastadaptivealphatest_beta_2.0","fastadaptivealphatestfixed_alpha_0.3_beta_2.0","fastadaptivealphatestfixed_alpha_0.5_beta_2.0","fastadaptivealphatestfixed_alpha_0.7_beta_2.0"]#,"fastadaptivealphatestfixed_alpha_1.0_beta_2.0"] # "adaptivealphatest_beta_2.0","adaptivealphatestid_beta_2.0"]
    t = 29
    # plot_fair_metrics(ds=ds,models=models,t=t)
    plot_fair_metrics_v2(ds=ds,models=models,t=t)
    plot_utility_metrics(ds=ds,models=models[1:],t=t)
    # get_models_vs_indegree(ds=ds,models=models,t=t)
    # get_avg_metric_all(models, ds, metric_="clustering", t=29)

    # get_statistical_imparity_all(models[1:], use_syn_ds=False, hMM=0.0, hmm=0.0, ds="rice")



