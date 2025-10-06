import torch
from typing import List, Tuple
from layers.egcnh import EGCUH


class EvolveGCN(torch.nn.Module):
    def __init__(self, layers: List[EGCUH]):
        super(EvolveGCN, self).__init__()
        self.layers_ = torch.nn.ModuleList(layers)

    def forward(self, adj: torch.Tensor, nodes: torch.Tensor, weights: torch.Tensor):
        new_weights = []
        for i in range(len(self.layers_)):
            nodes, nw = self.layers_[i]((adj, nodes, weights[i]))
            new_weights.append(nw)
        return nodes, new_weights

    def get_initial_weights(self, input_shape) -> List[torch.Tensor]:
        states = []
        s = input_shape
        for l in self.layers_:
            s = l.get_initial_weights(s)
            states.append(s)
            s = s.shape
        return states