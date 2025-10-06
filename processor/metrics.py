from typing import Sequence
import numpy as np
from scipy import sparse as sp


def modularity(adjacency: sp.spmatrix, clusters: Sequence[int]) -> float:
    """Compute graph modularity for an undirected graph.

    Args:
        adjacency: SciPy sparse adjacency matrix (any sparse format).
        clusters: Iterable of length N with integer cluster labels per node.

    Returns:
        Modularity value as a float.
    """
    if not sp.issparse(adjacency):
        adjacency = sp.csr_matrix(adjacency)
    elif not sp.isspmatrix_csr(adjacency):
        adjacency = adjacency.tocsr()

    clusters = np.asarray(clusters)
    degrees = adjacency.sum(axis=0).A1
    m2 = degrees.sum()  # equals 2 * (number of edges)
    if m2 == 0:
        return 0.0

    result = 0.0
    for cluster_id in np.unique(clusters):
        idx = np.where(clusters == cluster_id)[0]
        adj_sub = adjacency[idx][:, idx]
        deg_sub_sum = degrees[idx].sum()
        result += float(adj_sub.sum()) - (deg_sub_sum ** 2) / m2

    return float(result / m2)


def conductance(adjacency: sp.spmatrix, clusters: Sequence[int]) -> float:
    """Compute average conductance across clusters.

    Matches the computation style used in the original codebase:
    intra = sum of edges across the boundary of each cluster
    inter = sum of edges inside clusters
    conductance = intra / (inter + intra)

    Args:
        adjacency: SciPy sparse adjacency matrix (any sparse format).
        clusters: Iterable of length N with integer cluster labels per node.

    Returns:
        Average conductance value as a float in [0, 1].
    """
    if not sp.issparse(adjacency):
        adjacency = sp.csr_matrix(adjacency)
    elif not sp.isspmatrix_csr(adjacency):
        adjacency = adjacency.tocsr()

    clusters = np.asarray(clusters)
    n = adjacency.shape[0]
    all_nodes = np.arange(n)

    inter = 0.0  # within-cluster edges
    intra = 0.0  # across-cluster edges

    for cluster_id in np.unique(clusters):
        idx = np.where(clusters == cluster_id)[0]
        comp = np.setdiff1d(all_nodes, idx, assume_unique=True)
        adj_rows = adjacency[idx, :]
        inter += float(adj_rows[:, idx].sum())
        intra += float(adj_rows[:, comp].sum())

    denom = inter + intra
    if denom == 0:
        return 0.0
    return float(intra / denom)


