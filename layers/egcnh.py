import torch
import torch.nn as nn
from typing import Tuple
from .hgru import HGRUCell
from .summarize import SummarizeLayer


class EGCUH(nn.Module):
    def __init__(self, gru_cell: HGRUCell, summarize: SummarizeLayer, activation=None, dtype=torch.float32):
        super(EGCUH, self).__init__()

        self.gru_cell = gru_cell
        self.activation = getattr(torch.nn.functional, activation) if activation and hasattr(torch.nn.functional, activation) else torch.relu
        self.summarize = summarize
        self.dtype = dtype

    def forward(self, adj: torch.Tensor, nodes: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Convert sparse tensor to dense if needed
        if torch.is_sparse(adj):
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
            
        node_summary = self.summarize([nodes, weights.shape[-1]])
        weights_new = self.gru_cell([node_summary.t(), weights])

        # Matrix multiplication with adjacency matrix
        nodes_new = torch.matmul(adj_dense, nodes)
        nodes_new = self.activation(torch.matmul(nodes_new, weights_new))
        
        return nodes_new, weights_new

    def get_initial_weights(self, input_shape) -> torch.Tensor:
        return self.gru_cell.get_initial_state(input_shape)