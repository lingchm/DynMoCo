import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import matmul

class GCNConv(nn.Module):
    """Graph Convolutional Network layer in PyTorch."""

    def __init__(self, in_features, out_features, activation='selu', skip_connection=True, no_features=False):
        """
        Args:
            in_features: Input feature dimension.
            out_features: Output dimensionality of the layer.
            activation: Activation function to use for the final representations.
            skip_connection: If True, node features are propagated without neighborhood aggregation.
            no_features: If True, no feature transformation is applied.
        """
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.skip_connection = skip_connection
        self.no_features = no_features

        # Define kernel
        self.kernel = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

        if self.skip_connection:
            self.skip_weight = nn.Parameter(torch.ones(out_features))
        else:
            self.skip_weight = None

        # Set activation
        if activation == 'selu':
            self.activation = F.selu
        elif activation is None:
            self.activation = lambda x: x
        else:
            raise ValueError(f'GCN activation {activation} not supported')

    def forward(self, features, graph):
        """
        Args:
            features: (n, d) node feature matrix
            graph: (n, n) sparse graph adjacency matrix (e.g., torch_sparse.SparseTensor)
        Returns:
            (n, out_features) node representation matrix.
        """
        if self.no_features:
            output = self.kernel.weight.t()
            output = output.expand(features.shape[0], -1)
        else:
            output = self.kernel(features)

        if self.skip_connection:
            skip_term = output * self.skip_weight
        else:
            skip_term = 0

        output = skip_term + matmul(graph, output) + self.bias
        return self.activation(output)
    