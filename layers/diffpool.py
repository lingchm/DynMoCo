"""PyTorch implementation of DiffPooling layer.

This is a PyTorch implementation of the DiffPooling layer, which is similar to DMoN
but without the modularity loss components.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from collections import defaultdict

class DiffPooling(nn.Module):
    """Implementation of DiffPooling layer in PyTorch.

    DiffPooling layer implementation that performs graph pooling 

    Attributes:
        n_clusters: Number of clusters in the model.
        dropout_rate: Dropout rate applied to the intermediate representations.
        do_unpooling: Parameter controlling whether to perform unpooling.
    """

    def __init__(self, n_clusters, link_regularization=1, ent_regularization=1, dropout_rate=0, do_unpooling=False, activation="softmax", gumbel_tau=0.5, normalize=True):
        """Initialize the DiffPooling layer.

        Args:
            n_clusters: Number of clusters in the model.
            dropout_rate: Dropout rate applied to the intermediate representations.
            do_unpooling: Parameter controlling whether to perform unpooling.
        """
        super(DiffPooling, self).__init__()
        self.n_clusters = n_clusters
        self.dropout_rate = dropout_rate
        self.link_regularization = link_regularization
        self.ent_regularization = ent_regularization
        self.do_unpooling = do_unpooling
        self.normalize = normalize
        self.activation = activation
        self.gumbel_tau = gumbel_tau
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.losses = defaultdict(float)

    def reset_parameters(self, input_dim):
        """Initialize parameters based on input dimension.

        Args:
            input_dim: Input feature dimension.
        """
        self.transform = nn.Linear(input_dim, self.n_clusters)
        # Orthogonal initialization for the transform layer
        nn.init.orthogonal_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)

    def forward(self, features, adjacency):
        """Perform DiffPooling clustering.

        Args:
            features: (n, d) node feature matrix
            adjacency: (n, n) sparse graph adjacency matrix

        Returns:
            A tuple (features, clusters) with (k, d) cluster representations and
            (n, k) cluster assignment matrix, where k is the number of clusters,
            d is the dimensionality of the input, and n is the number of nodes.
        """
        if hasattr(adjacency, 'sparse_sizes'):
            assert features.shape[0] == adjacency.sparse_sizes()[0]
        else:
            assert features.shape[0] == adjacency.shape[0]

        # Compute cluster assignments
        logits = self.transform(features)
        if self.dropout is not None:
            logits = self.dropout(logits)
        
        if self.activation == "gumbelhard": assignments = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)
        elif self.activation == "gumbelsoft": assignments = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False)
        else: assignments = F.softmax(logits)
        
        cluster_sizes = torch.sum(assignments, dim=0)  # Size [k]
        assignments_pooling = assignments / (cluster_sizes + 1e-8)  # Size [n, k]

        # Pool features
        features_pooled = torch.matmul(assignments_pooling.t(), features)
        # features_pooled = F.selu(features_pooled)
        adjacency_pooled = torch.matmul(torch.matmul(assignments_pooling.t(), adjacency.to_dense()), assignments_pooling)
        
        if self.do_unpooling:
            features_pooled = torch.matmul(assignments_pooling, features_pooled)
            
        link_loss = adjacency.to_dense() - torch.matmul(assignments_pooling, assignments_pooling.t())
        link_loss = torch.norm(link_loss, p=2)
        if self.normalize:
            link_loss = link_loss / adjacency.numel()

        ent_loss = (-assignments_pooling * torch.log(assignments_pooling + 1e-15)).sum(dim=-1).mean()
        
        # Store losses for backpropagation
        self.losses["link_loss"] = link_loss * self.link_regularization
        self.losses["ent_loss"] = ent_loss * self.ent_regularization

        return features_pooled, assignments
    
    def get_losses(self):
        """Return the losses for backpropagation."""
        return self.losses