"""
Python Implementation of Fairness-Aware PageRank algorithm
https://dl.acm.org/doi/10.1145/3442381.3450065

Algorithm - Personalised Pagerank version for 
The Neighborhood LFPR Algorithm (LFPR_N)

Python Implementation of FairPageRank
Code Snippets borrowed from - https://github.com/mberr/torch-ppr/blob/main/src/torch_ppr
"""
import logging
logger = logging.getLogger(__name__)

from typing import Any, Collection, Mapping, Optional, Union

import torch
from torch.nn import functional

from tqdm.auto import tqdm

DeviceHint = Union[None, str, torch.device]


def power_iteration(
    adj: torch.Tensor,
    x0: torch.Tensor,
    alpha: float = 0.05,
    max_iter: int = 1000,
    use_tqdm: bool = False,
    epsilon: float = 1.0e-04,
    device: DeviceHint = None,
) -> torch.Tensor:
    r"""
    Perform the power iteration.

    .. math::
        \mathbf{x}^{(i+1)} = (1 - \alpha) \cdot \mathbf{A} \mathbf{x}^{(i)} + \alpha \mathbf{x}^{(0)}

    :param adj: shape: ``(n, n)``
        the (sparse) adjacency matrix
    :param x0: shape: ``(n,)``, or ``(n, batch_size)``
        the initial value for ``x``.
    :param alpha: ``0 < alpha < 1``
        the smoothing value / teleport probability
    :param max_iter: ``0 < max_iter``
        the maximum number of iterations
    :param epsilon: ``epsilon > 0``
        a (small) constant to check for convergence
    :param use_tqdm:
        whether to use a tqdm progress bar
    :param device:
        the device to use, or a hint thereof, cf. :func:`resolve_device`

    :return: shape: ``(n,)`` or ``(n, batch_size)``
        the ``x`` value after convergence (or maximum number of iterations).
    """
    # send tensors to device
    print("[Pagerank Calculation - Batch Size] - ", x0.size(1))
    MAX_LIMIT = 5000
    device = "cuda"
    # x0 = x0.to(device=device)
    no_batch = x0.ndim < 2
    if no_batch:
        x0 = x0.unsqueeze(dim=-1)
    # power iteration
    x_old = x = x0
    x = x.to(device=device)
    beta = 1.0 - alpha
    progress = tqdm(range(max_iter), unit_scale=True, leave=False, disable=not use_tqdm)
    for i in progress:
        x = x.to(device=device)   
        start, end = 0, int(adj.size(0))
        while True:
                if start >= end: break
                end_lim = start + MAX_LIMIT
                if end_lim >= end: end_lim = end
                # print("[PageRank] Spanning nodes from : {} to  {}".format(start, end_lim))
             
                # calculate x = (1 - alpha) * A.dot(x) + alpha * x0
                x[start:end_lim, :] = torch.sparse.addmm(
                    # dense matrix to be added
                    x0[start:end_lim, :].to(device=device),
                    # sparse matrix to be multiplied
                    adj[start:end_lim, :].to(device=device),
                    # dense matrix to be multiplied
                    x,
                    # multiplier for added matrix
                    beta=alpha,
                    # multiplier for product
                    alpha=beta,
                )
          
                start += MAX_LIMIT

        # note: while the adjacency matrix should already be row-sum normalized,
        #       we additionally normalize x to avoid accumulating errors due to loss of precision
        x = functional.normalize(x, dim=0, p=1)
        x = x.cpu()
        # calculate difference, shape: (batch_size,)
        mask = (torch.linalg.norm(x - x_old, ord=float("+inf"), axis=0))  > epsilon
        if not mask.any():
            logger.debug(f"Converged after {i} iterations up to {epsilon}.")
            break
        x_old = x
    # for/else, cf. https://book.pythontips.com/en/latest/for_-_else.html
    logger.warning(f"No convergence after {max_iter} iterations with epsilon={epsilon}.")

    if no_batch:
        x = x.squeeze(dim=-1)
    return x

# def power_iteration(
#     adj: torch.Tensor,
#     x0: torch.Tensor,
#     alpha: float = 0.05,
#     max_iter: int = 2,
#     use_tqdm: bool = False,
#     epsilon: float = 1.0e-04,
#     device: DeviceHint = None,
# ) -> torch.Tensor:
#     r"""
#     Perform the power iteration.

#     .. math::
#         \mathbf{x}^{(i+1)} = (1 - \alpha) \cdot \mathbf{A} \mathbf{x}^{(i)} + \alpha \mathbf{x}^{(0)}

#     :param adj: shape: ``(n, n)``
#         the (sparse) adjacency matrix
#     :param x0: shape: ``(n,)``, or ``(n, batch_size)``
#         the initial value for ``x``.
#     :param alpha: ``0 < alpha < 1``
#         the smoothing value / teleport probability
#     :param max_iter: ``0 < max_iter``
#         the maximum number of iterations
#     :param epsilon: ``epsilon > 0``
#         a (small) constant to check for convergence
#     :param use_tqdm:
#         whether to use a tqdm progress bar
#     :param device:
#         the device to use, or a hint thereof, cf. :func:`resolve_device`

#     :return: shape: ``(n,)`` or ``(n, batch_size)``
#         the ``x`` value after convergence (or maximum number of iterations).
#     """
#     # send tensors to device
#     device = "cpu"
#     adj = adj.to(device=device)
#     x0 = x0.to(device=device)
#     no_batch = x0.ndim < 2
#     if no_batch:
#         x0 = x0.unsqueeze(dim=-1)
#     # power iteration
#     x_old = x = x0
#     beta = 1.0 - alpha
#     progress = tqdm(range(max_iter), unit_scale=True, leave=False, disable=not use_tqdm)
#     for i in progress:
#         # calculate x = (1 - alpha) * A.dot(x) + alpha * x0
#         x = torch.sparse.addmm(
#             # dense matrix to be added
#             x0,
#             # sparse matrix to be multiplied
#             adj,
#             # dense matrix to be multiplied
#             x,
#             # multiplier for added matrix
#             beta=alpha,
#             # multiplier for product
#             alpha=beta,
#         )
#         # note: while the adjacency matrix should already be row-sum normalized,
#         #       we additionally normalize x to avoid accumulating errors due to loss of precision
#         print("shape of x after addmm operation: ", x.size(), x0.size())
#         x = functional.normalize(x, dim=0, p=1)
#         # calculate difference, shape: (batch_size,)
#         print("shape of x after functional operation: ", x.size())
#         diff = torch.linalg.norm(x - x_old, ord=float("+inf"), axis=0)
#         mask = diff > epsilon
#         if use_tqdm:
#             progress.set_postfix(
#                 max_diff=diff.max().item(), converged=1.0 - mask.float().mean().item()
#             )
#         if not mask.any():
#             logger.debug(f"Converged after {i} iterations up to {epsilon}.")
#             break
#         x_old = x
#     else:  # for/else, cf. https://book.pythontips.com/en/latest/for_-_else.html
#         logger.warning(f"No convergence after {max_iter} iterations with epsilon={epsilon}.")
#     print("x before squeeze: ", x.size())
#     if no_batch:
#         x = x.squeeze(dim=-1)
#     print("x returned shape: ", x.size())
#     return x



def prepare_x0(
    x0: Optional[torch.Tensor] = None,
    indices: Optional[Collection[int]] = None,
    n: Optional[int] = None,
) -> torch.Tensor:
    """
    Prepare a start value.

    The following precedence order is used:

    1. an explicit start value, via ``x0``. If present, this tensor is passed through without further modification.
    2. a one-hot matrix created via ``indices``. The matrix is of shape ``(n, len(indices))`` and has a single 1 per
       column at the given indices.
    3. a uniform ``1/n`` vector of shape ``(n,)``

    :param x0:
        the start value.
    :param indices:
        a non-zero indices
    :param n:
        the number of nodes

    :raises ValueError:
        if neither ``x0`` nor ``n`` are provided

    :return: shape: ``(n,)`` or ``(n, batch_size)``
        the initial value ``x``
    """
    if x0 is not None:
        return x0
    if n is None:
        raise ValueError("If x0 is not provided, n must be given.")
    if indices is not None:
        k = len(indices)
        x0 = torch.zeros(n, k)
        x0[indices, torch.arange(k, device=x0.device)] = 1.0 
        return x0
    return torch.full(size=(n,), fill_value=1.0 / n)

def get_fair_transition_matrices(adj, node_attr, psi):
    """
    R == 0
    B == 1
    P_N = psi*P_R + (1-psi)*P_B
    """
    one_minus_psi = (1-psi)
    P = torch.zeros_like(adj)
    groups_all, counts_all = torch.unique(node_attr, return_counts=True, sorted=True)
    group2frac = dict()
    for group, count in zip(groups_all, counts_all):
        group2frac[int(group)] = 1/count

    MAX_LIMIT = 5000
    start = 0
    end = int(adj.size(0))
    while True:
        if start >= end: break
        end_lim = start + MAX_LIMIT
        if end_lim >= end: end_lim = end
        print("[P Calculation] Spanning nodes from : {} to  {}".format(start, end_lim))
        adj_sub = adj[start:end_lim, :]


        mask = (adj_sub == 1)
        neighbors = mask.nonzero()[:, 1]
        degrees = mask.sum(dim=1)
        

        n_mask = (adj_sub == 0)
        n_neighbors = n_mask.nonzero()[:, 1]
        n_degrees = n_mask.sum(dim=1)
  
    
        start_index, n_start_index = 0, 0
        for index, node_group in enumerate(node_attr[start:end_lim]):
            size_ngh = int(degrees[index])
            nghs = neighbors[start_index:start_index+size_ngh]
            start_index += size_ngh
        

            n_size_ngh = int(n_degrees[index])
            n_nghs = n_neighbors[n_start_index:n_start_index+n_size_ngh]
            n_start_index += n_size_ngh
        
            # (1) get groups of neighbors
            group_of_nghs = node_attr[nghs].squeeze(1)
            neighborhood_groups, counts = torch.unique(group_of_nghs, return_counts=True, sorted=True)
            
        
            # (2) Populate transition matrices for neighborhood groups
            for ngh_group, degree_group in zip(neighborhood_groups, counts):
                
                nghs_of_that_group = nghs[(group_of_nghs == ngh_group).nonzero().squeeze(1)]
                if ngh_group == 0:
                    P[start+index, nghs_of_that_group] = psi/degree_group
                elif ngh_group == 1:
                    P[start+index, nghs_of_that_group] = one_minus_psi/degree_group
        
            # (3) Populate transition matrices for non-neighborhood groups
            non_neighborhood_groups = groups_all[~torch.isin(groups_all, neighborhood_groups)]
            n_group_of_nghs = node_attr[n_nghs].squeeze(1)
            for non_ngh_group in non_neighborhood_groups:
                n_nghs_of_that_group = n_nghs[(n_group_of_nghs == non_ngh_group).nonzero().squeeze(1)]
                if non_ngh_group == 0:
                    P[start+index, n_nghs_of_that_group] = psi*group2frac[int(non_ngh_group)]
                elif non_ngh_group == 1:
                    P[start+index, n_nghs_of_that_group] = (1-psi)*group2frac[int(non_ngh_group)]
                       
        start += MAX_LIMIT
   

    return P
                 
    

def get_fair_adjacency_matrix(adj: torch.Tensor, 
                              node_attr: torch.Tensor,
                              psi: Optional[float] = None) -> torch.Tensor:

    """
    The Neighborhood LFPR Algorithm
    """
     
    # (1. ) Compute P_R and P_B matrices
    P_N = get_fair_transition_matrices(adj, node_attr, psi)
    return P_N


def fair_personalized_page_rank(
    adj: torch.Tensor,
    node_attr: torch.Tensor,
    indices: torch.Tensor,
    test_edges: list,
    psi: float = 0.5, 
    **kwargs,
) -> torch.Tensor:
    """
    Batch-wise PPR computation with automatic memory optimization.

    :param adj: shape: ``(n, n)``
        the adjacency matrix.
    :param indices: shape: ``k``
        the indices for which to compute PPR
    :param kwargs:
        additional keyword-based parameters passed to :func:`power_iteration`

    :return: shape: ``(n, k)``
        the PPR vectors for each node index
    """
    batch_size = 10000
    results = list()
    
    P_N = get_fair_adjacency_matrix(adj=adj, node_attr=node_attr, psi=psi)

    if test_edges is not None:
        test_scores = torch.zeros(len(test_edges))
        test_us = torch.tensor([u for u, v in test_edges])
        test_vs = torch.tensor([v for u, v in test_edges])
   
    adj.fill_diagonal_(1)
    
    for indices_batch in torch.split(indices, batch_size):
        ppr_scores = power_iteration(adj=P_N, x0=prepare_x0(indices=indices_batch, n=adj.shape[0]), **kwargs).t()
        
        # non connected neighbors ppr scores

        adj[indices_batch] = adj[indices_batch].masked_fill(adj[indices_batch] == 1.0, -1000.0)
        adj[indices_batch] += ppr_scores

        max_ppr_score_nodes = torch.argmax(adj[indices_batch], dim=1)
        tensor_combined = torch.stack((indices_batch, max_ppr_score_nodes), dim=1)

        # Convert the result to a list of tuples
        list_combined = tensor_combined.tolist() 
        results.extend(list_combined)

        # find ppr scores for test
        if test_edges is not None:
            n_to_indices = {val.item(): idx for idx, val in enumerate(indices_batch)}
            # Use the map to get indices for elements in ppr
            ppr_indices = torch.tensor([n_to_indices[elem.item()] for elem in test_us if elem in indices_batch])          
            u_indices = torch.nonzero(torch.isin(test_us, indices_batch)).squeeze()
            vs = test_vs[u_indices]
            test_scores[u_indices] = ppr_scores[ppr_indices, vs]

 
    return results, test_scores if test_edges else None