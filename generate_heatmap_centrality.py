import os
import pandas as pd
import numpy as np
import seaborn as sns
import pickle as pkl
import argparse
import networkx as nx

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

from org.gesis.lib.n2v_utils import read_graph

main_path = "../AdaptiveAlpha/"

# import matplotlib as mpl
# fm = 0.3
N = 1000
YM, Ym = 2.5, 2.5
d = 0.03
topk = 10 # to extract top k 
hMM_list, hmm_list = np.arange(0,1.1,0.1), np.arange(0,1.1,0.1)

lim_dict = {
    "0.1" : {"llim":0,"ulim":0.011},
    "0.2" : {"llim":0,"ulim":0.006},
    "0.3" : {"llim":0,"ulim":0.004},
    "0.4" : {"llim":0,"ulim":0.0035},
}

plot_directory = "./plots/heatmap/centrality"
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

def get_diff_grid(files,model,centrality,group):
    dict_folder = main_path+"centrality/{}/{}".format(centrality,model)
    if not os.path.exists(dict_folder): os.makedirs(dict_folder)
  
    grid = np.zeros((len(hmm_list),len(hMM_list)))
    for file_name in files:
        # fm_ext = float(file_name.split("fm")[-1].split("-")[0])
        # if fm_ext != fm: continue
        hMM, hmm = file_name.split("hMM")[-1].split("-")[0], file_name.split("hmm")[-1].split("-")[0]
        hMM, hmm = hMM.replace(".gpickle","").replace("_t_29",""), hmm.replace(".gpickle","").replace("_t_29","")
        hMM_idx, hmm_idx = int(float(hMM)*10), int(float(hmm)*10)

        g = read_graph(file_name)
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
               

        centrality_1 = [val for node, val in centrality_dict.items() if g.nodes[node]["m"] == group]
        val_1 = np.mean(centrality_1)

        centrality_2 = [val for node, val in centrality_dict.items() if g.nodes[node]["m"] != group]
        val_2 = np.mean(centrality_2)
        diff = (val_1-val_2)
        grid[hmm_idx][hMM_idx] = diff
        # if hMM == "0.8" and hmm == "0.2":
        #      print("hMM: {}, hmm: {}, betn: {}".format(hMM, hmm,diff))
        # if hMM == "0.2" and hmm == "0.8":
        #      print("hMM: {}, hmm: {}, betn: {}".format(hMM, hmm,diff))
    return grid


def get_grid(files, model, centrality, group=1):
    dict_folder = main_path+"centrality/{}/{}".format(centrality,model)
    if not os.path.exists(dict_folder): os.makedirs(dict_folder)
  
    grid = np.zeros((len(hmm_list),len(hMM_list)))
    for file_name in files:
        # fm_ext = float(file_name.split("fm")[-1].split("-")[0])
        # if fm_ext != fm: continue
        
        hMM, hmm = file_name.split("hMM")[-1].split("-")[0], file_name.split("hmm")[-1].split("-")[0]
        hMM, hmm = hMM.replace(".gpickle","").replace("_t_29",""), hmm.replace(".gpickle","").replace("_t_29","")
        hMM_idx, hmm_idx = int(float(hMM)*10), int(float(hmm)*10)
       
        g = read_graph(file_name)

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
               

        cent_group = [val for node, val in centrality_dict.items() if g.nodes[node]["m"] == group]
        avg_val = np.mean(cent_group)
 
        grid[hmm_idx][hMM_idx] = avg_val
    return grid

    
def generate_heatmap(file_path, model, reco_type, centrality, diff=False, group=1, is_display=False):
    all_files = os.listdir(file_path)
    if "fm" in model and reco_type == "after":
         graph_files = [os.path.join(file_path,file_name) for file_name in all_files if "netmeta" not in file_name and ".gpickle" in file_name and "t_29" in file_name]
    else:
        graph_files = [os.path.join(file_path,file_name) for file_name in all_files if "netmeta" not in file_name and ".gpickle" in file_name]
    if diff:
        grid = get_diff_grid(graph_files, model, centrality, group)
    else:
        grid = get_grid(graph_files, model, centrality, group)
    print("No of files read: ", len(graph_files))
    if reco_type == "before":
        heatmap = grid.T
    elif reco_type == "after":       
        heatmap = grid.T 
        print(np.where(heatmap==np.max(heatmap)),np.where(heatmap==np.min(heatmap)), np.where(np.abs(heatmap)==np.min(np.abs(heatmap))))
      

    hmm_ticks = [np.round(hmm,2) for hmm in hmm_list]
    hMM_ticks = [np.round(hMM,2) for hMM in hMM_list]
    if centrality == "betweenness":
        fm = model.split("fm_")[-1].replace("_imp","").strip()
        if "n2v" in model: fm = "0.3"
        llim, uplim = lim_dict[fm].get("llim",0), lim_dict[fm].get("ulim",0.004)
  
        if diff: vmin, vmax = -uplim, uplim
        else: vmin, vmax = llim, uplim
    else:
        vmin, vmax = 0.0, 0.5
    if diff: cmap = plt.cm.coolwarm
    else: cmap = plt.cm.get_cmap('OrRd') 

    print("vmin:{}, vmax:{}".format(np.min(heatmap), np.max(heatmap)))  
    # vmin, vmax = np.round(np.min(heatmap),5),-np.round(np.min(heatmap),5)
    # ax = sns.heatmap(heatmap, cmap=plt.cm.coolwarm,xticklabels=hmm_ticks,yticklabels=hMM_ticks,vmin=vmin,vmax=vmax,cbar_kws={'ticks': [vmin,0,vmax]})
    # ax.collections[0].colorbar.set_ticklabels([str(vmin)+",Neg Variatn", "0", str(vmax)+" ,Pos Variatn"])
    
    # vmin, vmax = -0.002, 0.002
    # cmap =  plt.cm.coolwarm
    heatmap = heatmap[:-1,:-1]
    hmm_ticks, hMM_ticks = hmm_ticks[:-1], hMM_ticks[:-1]
    if is_display:
        ax = sns.heatmap(heatmap, cmap=cmap,xticklabels=hmm_ticks,yticklabels=hMM_ticks,vmin=vmin,vmax=vmax)

        
        ax.invert_yaxis()

        ax.set_xlabel("Homophily for Minority Class "+ r"($h_{mm}$)")
        ax.set_ylabel("Homophily for Majority Class "+ r"($h_{MM}$)")

        fig = ax.get_figure()
        fig.savefig(plot_directory+"/{}_{}_{}_diff_{}_group_{}.pdf".format(reco_type,model,centrality,diff,group),bbox_inches='tight')
        ax.clear()
    return heatmap

def generate_diff_heatmap(h1,h2):
    hmm_ticks = [np.round(hmm,2) for hmm in hmm_list]
    hMM_ticks = [np.round(hMM,2) for hMM in hMM_list]
    hmm_ticks, hMM_ticks = hmm_ticks[:-1], hMM_ticks[:-1]
    cmap = plt.cm.coolwarm
    diff = h1-h2
    vmin, vmax = -0.002, 0.002
    ax = sns.heatmap(diff, cmap=cmap,xticklabels=hmm_ticks,yticklabels=hMM_ticks,vmin=vmin,vmax=vmax)

    
    ax.invert_yaxis()

    ax.set_xlabel("Homophily for Minority Class "+ r"($h_{mm}$)")
    ax.set_ylabel("Homophily for Majority Class "+ r"($h_{MM}$)")

    fig = ax.get_figure()

    
    fig.savefig(plot_directory+"/_diff_group.pdf",bbox_inches='tight')

def compare_heatmap(heatmap1, heatmap2, model1, model2):

    compare = np.abs(heatmap2) < np.abs(heatmap1)
    val =  np.sum(compare)
    totalconfigs = heatmap1.shape[0]*heatmap1.shape[1]
    print("{} has improvement over {} in {}/{} configs".format(model2,model1,val,totalconfigs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model", type=str, default='.')
    parser.add_argument("--reco", help="before/after recommendation", type=str, default='')
    parser.add_argument("--centrality", help="closeness/betweenness", type=str, default='betweenness')
    parser.add_argument('--group', help="Minority (1) or Majority (0) Group", type=int, default=1)
    parser.add_argument('--diff', action='store_true')
    args = parser.parse_args()
    path = main_path+"{}".format(args.model)
    generate_heatmap(path, args.model, args.reco, args.centrality, args.diff, args.group, is_display=True) 

    #  ffw_p_1.0_q_1.0_fm_0.3, adaptivealphatest_beta_2.0_fm_0.3
    # model1, model2 = "fastadaptivealphatestfixed_alpha_0.7_beta_2.0_fm_0.3", args.model
    # path1, path2 = main_path+"{}".format(model1), main_path+"{}".format(model2)
    # h1 = generate_heatmap(path1, model1, args.reco, args.centrality, args.diff, args.group, is_display=False)
    # h2 = generate_heatmap(path2, model2, args.reco, args.centrality, args.diff, args.group, is_display=False)
    # compare_heatmap(h1,h2, model1, model2)
    # generate_diff_heatmap(h1,h2)

    