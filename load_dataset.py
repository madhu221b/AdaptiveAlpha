import os
import numpy as np
import networkx as nx
import pandas as pd
from facebook_scrap import get_graph
# from facebook_scrap import get_graph_syn
from org.gesis.lib.io import save_gpickle, read_pickle
from littleballoffur.exploration_sampling import ForestFireSampler

def get_edge_info(g):
    node_attrs = nx.get_node_attributes(g, "group")
    grps = list(set(node_attrs.values()))
    for grp in grps:
        print(grp, ": ", len([_ for node, _ in node_attrs.items() if _ == grp]))


    count_dict = dict()
    for edge in g.edges():
        u_id, v_id = node_attrs[edge[0]], node_attrs[edge[1]]
        label = "{}->{}".format(u_id,v_id)
        if label not in count_dict: count_dict[label] = 0
        count_dict[label] += 1

    print(count_dict)

def load_synth():
    dataset_path = "./data/synth2"
    node_attr_file = [os.path.join(dataset_path,filename) for filename in os.listdir(dataset_path) if ".attr" in filename][0]
    edge_file = [os.path.join(dataset_path,filename) for filename in os.listdir(dataset_path) if ".links" in filename][0]
    
    node_data, edge_data = list(), list()
    with open(node_attr_file, 'r') as fin:
        for line in fin:
            s = line.split()
            item = (s[0], {"group":int(s[1])})
            node_data.append(item)
    
    with open(edge_file, 'r') as fin:
        for line in fin:
            u,v = line.split()
            edge_data.append((u,v))
            


    print(len(node_data), len(edge_data))
    g = nx.DiGraph()
    g.add_nodes_from(node_data)
    g.add_edges_from(edge_data)
    return g

# def load_synth():
#     g = nx.powerlaw_cluster_graph(n=1000, m=15, p=0.05, seed=42)

#     node_data = list()

#     for node in g.nodes():
#         id_ = np.random.choice([0,1],size=1,p=[0.3,0.7])[0]
#         item = (node, {"group":id_})
#         node_data.append(item)



#     print(len(node_data))
#     g.add_nodes_from(node_data)
#     g = g.to_directed()
#     return g

def load_rice():
    """
    Group 0: Age is 18 or 19
    Group 1: Age is 20
    """
    dataset_path = "./data/rice"
    node_attr_file = os.path.join(dataset_path,"rice_subset.attr")
    edge_file = os.path.join(dataset_path,"rice_subset.links")
    
    node_data, edge_data = list(), list()
    mapping = dict()
    with open(node_attr_file, 'r') as fin:
        for i, line in enumerate(fin):
            s = line.split()
            item = (s[0], {"group":int(s[1])})
            node_data.append(item)
            mapping[s[0]] = i
    
    with open(edge_file, 'r') as fin:
        for line in fin:
            u,v = line.split()
            edge_data.append((u,v))
            


    print(len(node_data), len(edge_data))
    g = nx.DiGraph()
    g.add_nodes_from(node_data)
    g.add_edges_from(edge_data)
    nx.relabel_nodes(g, mapping, copy=False)
    save_gpickle(g, "./data/rice/rice_original.gpickle")
    print("isolated nodes: ", len(list(nx.isolates(g))))
    return g

def load_twitter():
    dataset_path = "./data/twitter"
    node_attr_file = os.path.join(dataset_path,"sample_4000.attr")
    edge_file = os.path.join(dataset_path,"sample_4000.links")
    
    node_data, edge_data = list(), list()
    mapping = dict()
    with open(node_attr_file, 'r') as fin:
        for i, line in enumerate(fin):
            s = line.split()
            item = (s[0], {"group":int(s[1])})
            node_data.append(item)
            mapping[s[0]] = i
    
    with open(edge_file, 'r') as fin:
        for line in fin:
            line = line.replace("[","").replace("]","").replace(",","").strip()
            u,v,_,_ = line.split()
            edge_data.append((u,v))
    print(len(node_data), len(edge_data))
    g = nx.DiGraph()
    g.add_nodes_from(node_data)
    g.add_edges_from(edge_data)
    # g.add_weighted_edges_from(edge_data)
    nx.relabel_nodes(g, mapping, copy=False)
    return g

def load_facebook(features=["gender"],syn_ds=False,n_comm=2):
    """
    Assuming only one attribute for now in features arr
    """
    if syn_ds and n_comm:
        g = get_graph_syn(n_comm)
    else:
        g = get_graph(features)
    return g   

def load_tuenti():
    path = "./data/tuenti/"
    node_file = path+"graph_nodes_by_gender.tsv"
    edge_file = path+"graph_edges_by_gender.tsv"
    
    node_df = pd.read_csv(node_file, sep = '\t')
    edge_df = pd.read_csv(edge_file, sep = '\t')
    
    node_df.set_index('user',inplace=True)
    node_df.rename(columns={"gender": "group"}, inplace=True)
    node_dict = node_df.to_dict('index')
    node_data = list(node_dict.items())

    edge_data = edge_df.apply(tuple, axis=1).tolist()
    
    g = nx.DiGraph()
    g.add_nodes_from(node_data)
    g.add_edges_from(edge_data)

    k = 0.15
    n = int(k*g.number_of_nodes())

    print("Sampling {} number of nodes from g".format(n))

    sampler = ForestFireSampler(number_of_nodes=n, seed= 42)
    g_sampled = sampler.sample(g.to_undirected())
    g = g.subgraph(list(g_sampled.nodes()))
    g_subset = g
    mapping = {node:i for i, node in enumerate(g_subset.nodes())}
    g = nx.relabel_nodes(g_subset, mapping)
    print("isolated nodes: ", len(list(nx.isolates(g))))
    return g

def load_pokec():
    path = "./data/pokec/"
    node_file = path+"graph_nodes_by_age.tsv"
    edge_file = path+"graph_edges_by_age.tsv"
    
    node_df = pd.read_csv(node_file, sep = '\t')
    edge_df = pd.read_csv(edge_file, sep = '\t')
    
    node_df.set_index('node',inplace=True)
    node_df.rename(columns={"age": "group"}, inplace=True)

    node_df.loc[node_df['group'] == "young", "group"] = 0
    node_df.loc[node_df['group'] == "old", "group"] = 1
    node_dict = node_df.to_dict('index')
    node_data = list(node_dict.items())

    edge_data = edge_df.apply(tuple, axis=1).tolist()
    
    g = nx.DiGraph()
    g.add_nodes_from(node_data)
    g.add_edges_from(edge_data)

    k = 0.15
    n = int(k*g.number_of_nodes())

    print("Sampling {} number of nodes from g".format(n))

    sampler = ForestFireSampler(number_of_nodes=n, seed= 42)
    g_sampled = sampler.sample(g.to_undirected())
    g = g.subgraph(list(g_sampled.nodes()))
    g_subset = g
    mapping = {node:i for i, node in enumerate(g_subset.nodes())}
    g = nx.relabel_nodes(g_subset, mapping)
    print("isolated nodes: ", len(list(nx.isolates(g))))
    return g

def load_dataset(name):
    if name == "rice":
        g = load_rice()
    elif name == "twitter":
        g = load_twitter()
    elif name == "synth2":
        g = load_synth()
    elif name == "tuenti":
        g = load_tuenti()
    elif name == "pokec":
        g = load_pokec()
    elif name.startswith("facebook"):
        if "locale" in name:
            g = load_facebook(features=["locale"])
        elif "syn" in name: # facebook_syn_2
            n_comm = int(name.split("syn_")[-1])
            g = load_facebook(syn_ds=True, n_comm=n_comm)
        elif "facebook100" in name:
            if "trinity" in name:
                g = nx.read_gpickle("./data/facebook100/facebook100_trinity.gpickle")
            else:
                g = nx.read_gpickle("./data/facebook100/facebook100_am.gpickle")
        else:
            g = load_facebook()
        
    return g

def load_twitter_climate():
    g = nx.read_gpickle("./data/twitter/twitter_climate.gpickle")
    return g
if __name__ == "__main__":
    g = load_dataset("rice")
    print(g.number_of_nodes(), g.number_of_edges())#, nx.average_clustering(g), nx.transitivity(g))
    get_edge_info(g)