"""PyTorch implementation of GCN with DiffPooling model from dmon/models/gcn_diffpool.py."""
import torch
import torch.nn as nn
from layers.gcn import GCNConv
from layers.diffpool import DiffPooling


class GCNDiffPool(nn.Module):
    """GCN with DiffPooling model in PyTorch.

    This is a PyTorch implementation of the GCN with DiffPooling model from the original TensorFlow code.
    """

    def __init__(self, architecture,  n_clusters, link_regularization=1, ent_regularization=1, dropout_rate=0):
        """Initialize the GCN with DiffPooling model.

        Args:
            channel_sizes: List of channel sizes for each GCN layer, with the last one being the number of clusters.
        """
        super(GCNDiffPool, self).__init__()
        self.architecture = architecture
        self.n_clusters = n_clusters
        self.dropout_rate = dropout_rate
        
        # Initialize GCN layers based on architecture
        self.gcn_layers = torch.nn.ModuleList()
        for in_dim, out_dim in zip(architecture[:-1], architecture[1:]):
            gcn_layer = GCNConv(in_dim, out_dim)  # Your corrected GCN class
            self.gcn_layers.append(gcn_layer)
            
        self.diffpool = DiffPooling(n_clusters, link_regularization, ent_regularization, dropout_rate, do_unpooling=False)
        self.diffpool.reset_parameters(architecture[-1])

    def forward(self, features, norm_adjacency, adjacency):
        output = features
        for gcn_layer in self.gcn_layers:
            output = gcn_layer(output, norm_adjacency)
        pool, pool_assignment = self.diffpool(output, adjacency)
        return pool, pool_assignment
    
    def get_losses(self):
        return self.diffpool.get_losses()