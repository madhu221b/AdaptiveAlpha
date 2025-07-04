import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from org.gesis.lib.DegFairGNN.models import DFair_GCN
from org.gesis.lib.DegFairGNN.layers import Discriminator
import org.gesis.lib.DegFairGNN.datasets as datasets
import datetime, time
import org.gesis.lib.DegFairGNN.utils
import argparse
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
import scipy.stats as st
import math

def compute_CI(out_list, name=None, log_file=None):
    ci = 1.96 * st.sem(out_list) / math.sqrt(len(out_list))
    log = name + ' Mean: {:.4f} '.format(np.mean(out_list)) + \
            'Std: {:.4f}'.format(st.sem(out_list)) 
    print(log)


#Get parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='syn', help='dataset')
parser.add_argument('--hMM', type=float, default=0.0, help='hMM')
parser.add_argument('--hmm', type=float, default=0.0, help='hmm')
parser.add_argument('--model', type=str, default='model', help='') # dummy placeholder
parser.add_argument('--start', type=float, default=0.0) # dummy placeholder
parser.add_argument('--end', type=float, default=0.0) # dummy placeholder
parser.add_argument('--name', type=str, default='name of ds', help='') # dummy placeholder
# model arguments
parser.add_argument('--d', type=int, default=1, help='degree evaluation')
parser.add_argument('--dim', type=int, default=32, help='hidden layer dimension')
parser.add_argument('--dim_d', type=int, default=32, help='degree mat dimension')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout percentage')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--omega', type=float, default=0.1, help='weight bias')
parser.add_argument('--k', type=float, default=1, help='ratio split head and tail group')

parser.add_argument('--w_b', type=float, default=1e-04, help='weight constraint')
parser.add_argument('--w_film', type=float, default=1e-04, help='weight FILM')
parser.add_argument('--w_f', type=float, default=1e-04, help='weight fair')
parser.add_argument('--decay', type=float, default=1e-04, help='weight decay')

# training arguments
parser.add_argument('--epochs', type=int, default=500, help='number of iteration')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
args = parser.parse_args()


cuda = torch.cuda.is_available()
print("cuda:", cuda)

class Controller(object):
    def __init__(self, model):
        self.model = model
        self.optim = optim.Adam(self.model.parameters(), 
                    lr=args.lr, weight_decay=args.decay)
       
     
    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def compute_loss(self, z, pos_edge_index, neg_edge_index):
        # Positive edge scores
        pos_scores = self.decode(z, pos_edge_index)
        pos_labels = torch.ones(pos_scores.size(0), device=z.device)

        # Negative edge scores
        neg_scores = self.decode(z, neg_edge_index)
        neg_labels = torch.zeros(neg_scores.size(0), device=z.device)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_labels, neg_labels])

        return F.binary_cross_entropy_with_logits(scores, labels)

    def train(self, data):
        print('Training for epochs: ', args.epochs)
        

        for i in range(args.epochs):
            self.model.train()
            self.optim.zero_grad()
            output, b, film = self.model(data.feat, data.adj, data.degree)               
            assert not torch.isnan(output).any()                         
            
            train_output = output
            mean = torch.mean(data.degree.float())

            idx_low = torch.where(data.degree < mean)[0]
            idx_high = torch.where(data.degree >= mean)[0]

            low_embed = torch.mean(train_output[idx_low], dim=0)
            high_embed = torch.mean(train_output[idx_high], dim=0)


            sp_loss = F.mse_loss(low_embed, high_embed)


            L_cls = self.compute_loss(output, data.pos_edge_index, data.neg_edge_index)

        
            loss = L_cls + (args.w_f * sp_loss) + (args.w_b * b) + (args.w_b * film) 
            loss.backward()
            self.optim.step()
        return


    def test(self, data):
        print('Testing ...')
        
        self.model.eval()
        print("eval state")
        
        # accuracy
        output, _, _ = self.model(data.feat, data.adj, data.degree)
        return output
        
        # preds = self.decode(output, pos_edge_index)
        # labels = np.ones(preds.shape)
        # neg_preds = torch.sigmoid(neg_score).cpu().detach().numpy()
        # neg_labels = np.zeros(neg_preds.shape)

        # all_preds = np.concatenate([preds, neg_preds])
        # all_labels = np.concatenate([labels, neg_labels])

        # auc = roc_auc_score(all_labels, all_preds)
        # acc = acc.cpu()
        # print('Accuracy={:.4f}'.format(acc))

        # return acc*100, 0, 0, 0


def main():
    print(str(args))

  
    np.random.seed(args.seed)
    

    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)


    data = datasets.get_dataset(args.dataset, hMM=args.hMM, hmm=args.hmm)
    in_feat = data.feat.size(1)


    model = DFair_GCN(args, in_feat, data.max_degree)


    if cuda:
        model.cuda()
        # data.to_cuda()
        
    controller = Controller(model)
    controller.train(data)

    # acc, macf, out1, out2 = controller.test(data)


def _get_node_embeddings_gnn(G, is_syn=False, name=""):
    data = datasets.get_dataset("syn" if is_syn else name, G)
    in_feat = data.feat.size(1)


    model = DFair_GCN(args, in_feat, data.max_degree)


    if cuda:
        model.cuda()
        data.to_cuda()
        
    controller = Controller(model)
    controller.train(data)
    print("training done")
    embeddings = controller.test(data)
    return embeddings


# if __name__ == "__main__":
#     main()
