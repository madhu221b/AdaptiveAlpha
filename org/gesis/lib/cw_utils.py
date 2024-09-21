import itertools
import pandas as pd
import numpy as np
import networkx as nx

def get_walks_by_src_nodes(walks):
    walk_dict = dict()
    for walk in walks:
        walk  = [int(_) for _ in walk]
        if walk[0] not in walk_dict: walk_dict[walk[0]] = list()
        walk_dict[walk[0]].append(walk)
    return walk_dict

def get_closeness_to_boundary(G, group_df, walk_dict, p, d, r):
    print("Calculating Closeness to Boundary..(Raising to p-power already)")
    m_dict = dict()
    denom = r*d
    for u in G.nodes():
        I_u = group_df.loc[u,"group"]
        W_u = walk_dict[u] # get walks which start with node u
        W_u = list(itertools.chain(*W_u)) # flatten all the walks
        df = pd.Series(W_u).to_frame(name="node")
        df["group"] = df.apply(lambda row: group_df.loc[row["node"],"group"], axis=1)
        df_mask = df[df["group"] != I_u]
        num = df_mask.shape[0]
        m_dict[u] = np.power(num/denom, p)

    m_df = pd.DataFrame.from_dict(m_dict, orient='index', columns=['proximity'])
    return m_df

def get_adj_matrix(g, weight_key="weight"):
    nodes = g.number_of_nodes()
    A = np.zeros((nodes,nodes))
    for u,v in g.edges():
        w = g[u][v].get("weight", 1)
        if type(u) is str: u = int(u)
        if type(v) is str: v = int(v)
        A[u][v] = w
    return A

def get_upweighted_weights(G, walks, p, alpha, d, r):
    weight_dict = dict()
    node_attrs = nx.get_node_attributes(G, "group")
    group_df = pd.DataFrame.from_dict(node_attrs, orient='index', columns=['group']) 
    A = get_adj_matrix(G)
    walk_dict = get_walks_by_src_nodes(walks)
    m_df = get_closeness_to_boundary(G, group_df, walk_dict, p, d, r)
    print("Upweighting the Edges ..")
    for v in G.nodes():
        nghs_df = pd.Series(G.successors(v)).to_frame(name="node")
        nghs_df["group"] = nghs_df.apply(lambda row: group_df.loc[row["node"],"group"], axis=1)
        I_v = group_df.loc[v,"group"] 
        N_v = list(nghs_df[nghs_df['group'] == I_v]["node"])
        R_v = nghs_df[nghs_df['group'] != I_v]["group"].unique()
         
        Z_same = np.dot(A[v,N_v],m_df.loc[N_v,"proximity"])
        if Z_same != 0:
            for u in N_v:
                val = (A[v,u]*(1-alpha)*m_df.loc[u,"proximity"])/Z_same
                weight_dict[(v,u)] = {"weight":val}
        
        for c in R_v:
            N_v_c = list(nghs_df[nghs_df['group'] == c]["node"])
            Z_diff = len(R_v)*np.dot(A[v,N_v_c],m_df.loc[N_v_c,"proximity"])
            if Z_diff != 0:
                for u in N_v_c:
                    if len(N_v) != 0:
                        val = (A[v,u]*(alpha)*m_df.loc[u,"proximity"])/Z_diff
                        weight_dict[(v,u)] = {'weight':val}
                    else:
                        val = (A[v,u]*m_df.loc[u,"proximity"])/Z_diff
                        weight_dict[(v,u)] = {'weight':val}


    return weight_dict