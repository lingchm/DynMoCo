"""Helper functions for graph processing in PyTorch."""
import numpy as np
import scipy.sparse
import torch
from torch_sparse import SparseTensor
import re



def normalize_graph(graph, normalized=True, add_self_loops=True):
    """Normalize the graph's adjacency matrix in the scipy sparse matrix format.

    Args:
        graph: A scipy sparse adjacency matrix of the input graph.
        normalized: If True, uses the normalized Laplacian formulation. Otherwise,
          use the unnormalized Laplacian construction.
        add_self_loops: If True, adds a one-diagonal corresponding to self-loops in
          the graph.

    Returns:
        A scipy sparse matrix containing the normalized version of the input graph.
    """
    if add_self_loops:
        graph = graph + scipy.sparse.identity(graph.shape[0])
    degree = np.squeeze(np.asarray(graph.sum(axis=1)))
    if normalized:
        with np.errstate(divide='ignore'):
            inverse_sqrt_degree = 1. / np.sqrt(degree)
        inverse_sqrt_degree[inverse_sqrt_degree == np.inf] = 0
        inverse_sqrt_degree = scipy.sparse.diags(inverse_sqrt_degree)
        return inverse_sqrt_degree @ graph @ inverse_sqrt_degree
    else:
        with np.errstate(divide='ignore'):
            inverse_degree = 1. / degree
        inverse_degree[inverse_degree == np.inf] = 0
        inverse_degree = scipy.sparse.diags(inverse_degree)
        return inverse_degree @ graph


def convert_scipy_sparse_to_sparse_tensor(matrix):
    """Converts a scipy sparse matrix to PyTorch SparseTensor.

    Args:
        matrix: A scipy sparse matrix.

    Returns:
        A PyTorch SparseTensor.
    """
    matrix = matrix.tocoo()
    indices = torch.LongTensor([matrix.row, matrix.col])
    values = torch.FloatTensor(matrix.data)
    return SparseTensor(row=indices[0], col=indices[1], value=values, 
                       sparse_sizes=(matrix.shape[0], matrix.shape[1]))


def load_npz(filename):
    """Loads an attributed graph with sparse features from a specified Numpy file.

    Args:
        filename: A valid file name of a numpy file containing the input data.

    Returns:
        A tuple (graph, features, labels, label_indices) with the sparse adjacency
        matrix of a graph, sparse feature matrix, dense label array, and dense label
        index array (indices of nodes that have the labels in the label array).
    """
    with np.load(open(filename, 'rb'), allow_pickle=True) as loader:
        loader = dict(loader)
        adjacency = scipy.sparse.csr_matrix(
            (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
            shape=loader['adj_shape'])

        features = scipy.sparse.csr_matrix(
            (loader['feature_data'], loader['feature_indices'],
             loader['feature_indptr']),
            shape=loader['feature_shape'])

        label_indices = loader['label_indices']
        labels = loader['labels']
    
    assert adjacency.shape[0] == features.shape[0], 'Adjacency and feature size must be equal!'
    assert labels.shape[0] == label_indices.shape[0], 'Labels and label_indices size must be equal!'
    return adjacency, features, labels, label_indices 