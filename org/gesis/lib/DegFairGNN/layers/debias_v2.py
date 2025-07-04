import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import scipy.sparse as sp
import numpy as np


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class Debias_v2(nn.Module):
    def __init__(self, args, in_feat, out_feat, d_max):
        super(Debias_v2, self).__init__()

        self.dim_M = args.dim_d
        self.out_feat = out_feat
        self.omega = args.omega
        self.d_max = (d_max+1) #0->dmax
        self.dataset = args.dataset
        self.k = args.k
        #self.w = args.w
        #self.sparse = args.sparse

        
        self.weight = nn.Linear(in_feat, out_feat)

        self.W_gamma = nn.Parameter(torch.FloatTensor(self.dim_M, out_feat))
        self.W_beta = nn.Parameter(torch.FloatTensor(self.dim_M, out_feat))
        self.U_gamma = nn.Parameter(torch.FloatTensor(out_feat, out_feat))
        self.U_beta = nn.Parameter(torch.FloatTensor(out_feat, out_feat))
        self.b_gamma = nn.Parameter(torch.FloatTensor(1, out_feat))
        self.b_beta = nn.Parameter(torch.FloatTensor(1, out_feat))

        self.W_add = nn.Linear(out_feat, out_feat, bias=False)
        self.W_rev = nn.Linear(out_feat, out_feat, bias=False)

        # Positional Encoding
        PE = np.array([
            [pos / np.power(10000, (i-i%2)/self.dim_M) for i in range(self.dim_M)]
            for pos in range(self.d_max)])

        PE[:, 0::2] = np.sin(PE[:, 0::2]) 
        PE[:, 1::2] = np.cos(PE[:, 1::2]) 
        self.PE = torch.as_tensor(PE, dtype=torch.float32)
        
        self.set_parameters()


    def set_parameters(self):
        #nn.init.uniform_(self.m)
        nn.init.uniform_(self.W_gamma)
        nn.init.uniform_(self.W_beta)
        nn.init.uniform_(self.U_gamma)
        nn.init.uniform_(self.U_beta)
        nn.init.uniform_(self.b_gamma)
        nn.init.uniform_(self.b_beta)

        '''
            M_stdv = 1. / math.sqrt(self.M.size(1))
            self.M.data.uniform_(-M_stdv, M_stdv)

            b_stdv = 1. / math.sqrt(self.b.size(1))
            self.b.data.uniform_(-b_stdv, b_stdv)

            for m in self.modules():
                print(m.weight)
        '''

    def forward(self, x, adj, degree, idx):
        h = self.weight(x)
        m_dv = torch.squeeze(self.PE[degree.cpu()])
        m_dv = m_dv.cuda()

        # version 1
        if self.dataset != 'nba':
            h *= self.dim_M**0.5
        gamma = F.leaky_relu(torch.matmul((m_dv), self.W_gamma) + self.b_gamma) #
        beta = F.leaky_relu(torch.matmul((m_dv), self.W_beta) + self.b_beta) #
        

        #neighbor mean
        
        i = torch.spmm(adj, h.cpu()).cuda()
        nonzero = degree != 0
        degree = degree.unsqueeze(1) 
        i[nonzero] = i[nonzero] / degree[nonzero]
        assert not torch.isnan(i).any()

        # debias low-degree
        b_add = (gamma + 1) * self.W_add(i) + beta

        # debias high-degree
        b_rev = (gamma + 1) * self.W_rev(i) + beta


        mean_degree = torch.mean(degree.float())
        K = mean_degree * self.k
        R = torch.where(degree < K, torch.cuda.FloatTensor([1.]), torch.cuda.FloatTensor([0.]))

        L_b = torch.sum(torch.norm((R*b_add)[idx], dim=1)) + torch.sum(torch.norm(((1-R)*b_rev)[idx], dim=1))
        L_b /= idx.shape[0]


        L_film = torch.sum(torch.norm(gamma[idx], dim=1)) + torch.sum(torch.norm(beta[idx], dim=1))
        L_film /= idx.shape[0]
     
        bias = self.omega * (R * b_add - (1-R) * b_rev)

        output = torch.mm(adj, h.cpu()).cuda() + h + bias
        output /= (degree + 1)


        return F.leaky_relu(output), L_b, L_film

    # def forward(self, x, adj, degree, idx):
    #     h = self.weight(x)
    #     m_dv = torch.squeeze(self.PE[degree.cpu()])
    #     m_dv = m_dv.cuda()

    #     # version 1
    #     if self.dataset != 'nba':
    #         h *= self.dim_M**0.5
    #     gamma = F.leaky_relu(torch.matmul((m_dv), self.W_gamma) + self.b_gamma) #
    #     beta = F.leaky_relu(torch.matmul((m_dv), self.W_beta) + self.b_beta) #
        
        
    #     chunk_size = 5000
    #     num_nodes = adj.size(0)

    #     outputs = []
    #     Rs = []
    #     b_adds, b_revs = [], []
    #     for start in range(0, num_nodes, chunk_size):
    #             end = min(start + chunk_size, num_nodes)
    #             # print(adj.device, adj[start:end, :].size())
    #             # adj_chunk = adj[start:end, :].cuda()       # Shape: [B, N]
    #             h_chunk = h                         # Shape: [N, D]
    #             # i_chunk = torch.mm(adj_chunk, h_chunk)  # Shape: [B, D]
    #             i_chunk = torch.spmm(adj, h_chunk.cpu()).cuda() 
    #             deg_chunk = degree[start:end].unsqueeze(1)  # [B, 1]
    #             nonzero = deg_chunk.view(-1) != 0
    #             i_chunk[nonzero] = i_chunk[nonzero] / deg_chunk[nonzero]
    #             assert not torch.isnan(i_chunk[nonzero]).any()
    #             # debias low-degre
    #             b_add = (gamma[start:end] + 1) * self.W_add(i_chunk) + beta[start:end]
    #             # debias high-degree
    #             b_rev = (gamma[start:end] + 1) * self.W_rev(i_chunk) + beta[start:end]

    #             mean_degree = degree.float().mean()
    #             K = mean_degree * self.k
    #             R = (degree[start:end] < K).float().unsqueeze(1)  # [B, 1]
        #         bias = self.omega * (R * b_add - (1 - R) * b_rev)

        #         out_chunk = i_chunk + h[start:end] + bias
        #         out_chunk /= (degree[start:end].unsqueeze(1) + 1)
 
        #         outputs.append(out_chunk)
        #         Rs.append(R)
        #         b_adds.append(b_add)
        #         b_revs.append(b_rev)
    


        # R = torch.cat(Rs, dim=0)
        # b_add = torch.cat(b_adds, dim=0)
        # b_rev = torch.cat(b_revs, dim=0)

        # L_b = torch.sum(torch.norm((R*b_add)[idx], dim=1)) + torch.sum(torch.norm(((1-R)*b_rev)[idx], dim=1))
        # L_b /= idx.shape[0]


        # L_film = torch.sum(torch.norm(gamma[idx], dim=1)) + torch.sum(torch.norm(beta[idx], dim=1))
        # L_film /= idx.shape[0]
     

        # output = torch.cat(outputs, dim=0) 
        
        # return F.leaky_relu(output), L_b, L_film

