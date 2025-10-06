"""PyTorch implementation of multilayer GCN model from dmon/models/multilayer_gcn.py."""
import torch
import torch.nn as nn
from layers.gcn import GCNConv


class MultilayerGCN(nn.Module):
    """Multilayer Graph Convolutional Network model in PyTorch.

    This is a PyTorch implementation of the multilayer GCN model from the original TensorFlow code.
    """

    def __init__(self, channel_sizes):
        """Initialize the multilayer GCN model.

        Args:
            channel_sizes: List of channel sizes for each GCN layer.
        """
        super(MultilayerGCN, self).__init__()
        self.channel_sizes = channel_sizes
        self.gcn_layers = nn.ModuleList()

    def forward(self, features, graph):
        """Forward pass of the multilayer GCN model.

        Args:
            features: (n, d) node feature matrix
            graph: (n, n) sparse graph adjacency matrix

        Returns:
            An (n, last_channel_size) node representation matrix.
        """
        output = features
        for i, n_channels in enumerate(self.channel_sizes):
            gcn_layer = GCNConv(n_channels)
            gcn_layer.reset_parameters(output.shape[1])
            output = gcn_layer(output, graph)
        return output


def multilayer_gcn(inputs, channel_sizes):
    """Create a multilayer GCN model.

    Args:
        inputs: Tuple of (features, graph)
        channel_sizes: List of channel sizes for each GCN layer

    Returns:
        A PyTorch model that applies multilayer GCN.
    """
    features, graph = inputs
    
    class MultilayerGCNModel(nn.Module):
        def __init__(self):
            super(MultilayerGCNModel, self).__init__()
            self.gcn_model = MultilayerGCN(channel_sizes)
            
        def forward(self, features, graph):
            return self.gcn_model(features, graph)
    
    return MultilayerGCNModel() 