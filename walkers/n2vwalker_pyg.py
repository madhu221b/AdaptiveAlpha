from collections import Counter
import networkx as nx
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import copy
import sys

from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx
import torch

try:
  from .walker import Walker
except Exception as error:
    from walker import Walker

class N2VWalker(Walker):
    def __init__(self, graph, p=1, q=1, workers=1, dimensions=64, walk_len=10, num_walks=200):
        print(f"N2V implementation by pytorch geometric with p:{p}, q:{q}")
        self.p, self.q = p, q
        super().__init__(graph, workers=workers, dimensions=dimensions, walk_len=walk_len, num_walks=num_walks)
        self.quiet = False
        self.number_of_nodes = self.graph.number_of_nodes()
 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data = from_networkx(graph)
        self.model = Node2Vec(
              self.data.edge_index,
              embedding_dim=128,
              walk_length=20,
              context_size=10,
              walks_per_node=10,
              num_negative_samples=1,
              p=self.p,
              q=self.q,
              sparse=True).to(self.device)

        
    def train(self):
        num_workers = 4 if sys.platform == 'linux' else 0
        loader = self.model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
        optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)

        self.model.train()

        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def get_embeddings(self):

        for epoch in range(1, 101):
            loss = self.train()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        with torch.no_grad():
          self.model.eval()
          z = self.model().cpu()
          return z










    

 



