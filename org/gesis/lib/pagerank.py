"""
Python Implementation of PageRank
Code Snippets borrowed from - https://github.com/mberr/torch-ppr/blob/main/src/torch_ppr
"""













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
        print("this is called")
        k = len(indices)
        x0 = torch.zeros(n, k)
        x0[indices, torch.arange(k, device=x0.device)] = 1.0
        return x0
    return torch.full(size=(n,), fill_value=1.0 / n)




def personalized_page_rank(
    adj: torch.Tensor,
    node_attr: torch.Tensor
    indices: torch.Tensor,
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
    batch_size = 1
    return torch.cat(
        [
            power_iteration(adj=adj, x0=prepare_x0(indices=indices_batch, n=adj.shape[0]), **kwargs)
            for indices_batch in torch.split(indices, batch_size)
        ],
        dim=1,
    )