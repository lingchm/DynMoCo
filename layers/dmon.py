
"""Deep Modularity Network (DMoN) PyTorch layer.

Deep Modularity Network (DMoN) layer implementation as presented in
"Graph Clustering with Graph Neural Networks" in PyTorch.
DMoN optimizes modularity clustering objective in a fully unsupervised regime.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict

class DMoNPooling(nn.Module):
    """Implementation of Deep Modularity Network (DMoN) layer.

    Deep Modularity Network (DMoN) layer implementation as presented in
    "Graph Clustering with Graph Neural Networks" in PyTorch.
    DMoN optimizes modularity clustering objective in a fully unsupervised mode,
    however, this implementation can also be used as a regularizer in a supervised
    graph neural network. Optionally, it does graph unpooling.

    Attributes:
        n_clusters: Number of clusters in the model.
        collapse_regularization: Collapse regularization weight.
        dropout_rate: Dropout rate. Note that the dropout is applied to the
          intermediate representations before the softmax.
        do_unpooling: Parameter controlling whether to perform unpooling of the
          features with respect to their soft clusters. If true, shape of the input
          is preserved.
    """

    def __init__(self, n_clusters, collapse_regularization=0.1, 
                 structural_regularization=0.0, knowledge_regularization=0.0, 
                 connectivity_regularization=0.0, sparsity_regularization=0.0, 
                 activation="softmax", gumbel_tau=0.5,
                 dropout_rate=0, do_unpooling=False, normalize=True):
        """Initializes the layer with specified parameters."""
        super(DMoNPooling, self).__init__()
        self.n_clusters = n_clusters
        self.collapse_regularization = collapse_regularization
        self.knowledge_regularization = knowledge_regularization
        self.sparsity_regularization = sparsity_regularization
        self.connectivity_regularization = connectivity_regularization
        self.gumbel_tau = gumbel_tau
        self.dropout_rate = dropout_rate
        self.do_unpooling = do_unpooling
        self.transform = None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.activation = activation
        self.normalize = normalize
        self.losses = defaultdict(float)
        
    def reset_parameters(self, input_dim, device='cpu', seed=123):
        """Initialize parameters based on input dimension.
        
        Args:
            input_dim: Input feature dimension
            adjacency: Adjacency matrix for Louvain initialization (optional)
            features: Node features for Louvain initialization (optional)
            use_louvain: Whether to use Louvain algorithm for initialization
            device: Device to place tensors on
            seed: Random seed for reproducible Louvain results
        """
        self.transform = nn.Linear(input_dim, self.n_clusters)
        # Use orthogonal initialization
        nn.init.orthogonal_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)
    
    def get_parameters(self):
        return self.transform.weight.data, self.transform.bias.data
    
    def set_parameters(self, weight, bias=None):
        self.transform.weight.data = weight
        if bias is not None:
            self.transform.bias.data = bias
    
    def link_prediction_loss(self, adjacency, similarity_assignments):
        link_loss = adjacency.to_dense() - similarity_assignments
        if self.normalize:
            link_loss = link_loss / adjacency.numel()
        return torch.norm(link_loss, p=2) 
    
    def knowledge_loss_negative(self, comembership, similarity_assignments):
        mask = (comembership == 0)
        masked_sim = similarity_assignments[mask]
        return torch.mean(masked_sim ** 2)
    
    def knowledge_loss_positive(self, comembership, assignments):
        i, j = (comembership > 0).nonzero(as_tuple=True)
        pairwise_loss = ((assignments[i] - assignments[j]) ** 2).sum(dim=1)  # shape: [num_pairs]
        return pairwise_loss.mean()

    def forward(self, features, adjacency, comembership=None, prior_membership=None):
        """Performs DMoN clustering according to input features and input graph.

        Args:
            features: (n, d) node feature matrix
            adjacency: (n, n) sparse graph adjacency matrix

        Returns:
            A tuple (features, clusters) with (k, d) cluster representations and
            (n, k) cluster assignment matrix, where k is the number of clusters,
            d is the dimensionality of the input, and n is the number of nodes in the
            input graph. If do_unpooling is True, returns (n, d) node representations
            instead of cluster representations.
        """
        assert len(features.shape) == 2
        # Use sparse_sizes() for SparseTensor objects
        if hasattr(adjacency, 'sparse_sizes'):
            assert features.shape[0] == adjacency.sparse_sizes()[0]
        else:
            assert features.shape[0] == adjacency.shape[0]
        
        # Compute cluster assignments
        logits = self.transform(features)
        if self.dropout is not None:
           logits = self.dropout(logits)

        if self.activation == "gumbelhard": 
            assignments = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)
        elif self.activation == "gumbelsoft": 
            assignments = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False)
        else: 
            assignments = F.softmax(logits)
    
        features_pooled = self.pool(features, adjacency, assignments, comembership, prior_membership)
        
        return features_pooled, assignments
    
    def pool(self, features, adjacency, assignments, comembership=None, prior_membership=None):
        
        cluster_sizes = torch.sum(assignments, dim=0)  # Size [k]
        assignments_pooling = assignments / (cluster_sizes + 1e-8)  # Size [n, k]
        
        # Compute degrees
        degrees = torch.sum(adjacency.to_dense(), dim=0)   # Size [n]
        degrees = degrees.unsqueeze(1)  # Size [n, 1]

        # Get number of nodes from sparse_sizes() if available
        if hasattr(adjacency, 'sparse_sizes'):
            number_of_nodes = adjacency.sparse_sizes()[1]
        else:
            number_of_nodes = adjacency.shape[1]
        number_of_edges = torch.sum(degrees)

        # Compute the size [k, k] pooled graph as S^T*A*S in two multiplications
        graph_pooled = torch.matmul(assignments.t(), matmul(adjacency, assignments))

        # Compute the rank-1 normalizer matrix S^T*d*d^T*S efficiently
        normalizer_left = torch.matmul(assignments.t(), degrees) # Left part is [k, 1] tensor
        normalizer_right = torch.matmul(degrees.t(), assignments) # Right part is [1, k] tensor  
        
        # Normalizer is rank-1 correction for degree distribution
        normalizer = torch.matmul(normalizer_left, normalizer_right) / (2 * number_of_edges)
        spectral_loss = -torch.trace(graph_pooled - normalizer) / (2 * number_of_edges)
        collapse_loss = torch.norm(cluster_sizes) / number_of_nodes * torch.sqrt(torch.tensor(float(self.n_clusters))) - 1
        similarity_assignments = torch.matmul(assignments, assignments.t())
        knowledge_loss = self.knowledge_loss_negative(comembership, similarity_assignments) if self.knowledge_regularization > 0 else torch.tensor(0.0)
        connectivity_loss = ((1 - adjacency.to_dense()) * similarity_assignments).mean() if self.connectivity_regularization > 0 else torch.tensor(0.0)
        sparsity_loss = torch.norm(prior_membership - assignments, p=1) if self.sparsity_regularization > 0 and prior_membership is not None else torch.tensor(0.0)
        
        # Store losses for backpropagation
        self.losses["spectral_loss"] = spectral_loss
        self.losses["collapse_loss"] = self.collapse_regularization * collapse_loss
        self.losses["knowledge_loss"] = self.knowledge_regularization * knowledge_loss
        self.losses["connectivity_loss"] = self.connectivity_regularization * connectivity_loss
        self.losses["sparsity_loss"] = self.sparsity_regularization * sparsity_loss

        # Pool features
        features_pooled = torch.matmul(assignments_pooling.t(), features)
        features_pooled = F.selu(features_pooled)
        
        if self.do_unpooling:
            features_pooled = torch.matmul(assignments_pooling, features_pooled)
            
        return features_pooled

    def get_losses(self):
        """Return the losses for backpropagation."""
        
        return self.losses
    