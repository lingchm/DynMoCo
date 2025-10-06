import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SummarizeLayer(nn.Module):
    def __init__(self, input_dim: int, dtype=torch.float32):
        super(SummarizeLayer, self).__init__()
        self.p = nn.Parameter(torch.empty(input_dim, dtype=dtype))
        # Use uniform initialization for 1D tensors instead of xavier_uniform_
        nn.init.uniform_(self.p, -0.1, 0.1)

    def forward(self, x: torch.Tensor, k: int) -> torch.Tensor:
        # x: (batch_size, input_dim) or (num_items, input_dim)
        # k: top-k value

        # Compute projection scores
        y = F.linear(x, self.p / self.p.norm())  # equivalent to tf.linalg.matvec(x, self.p / ||p||)

        # Get top-k values and indices
        top_y_vals, top_y_inds = torch.topk(y, k)

        # Gather the top-k rows of x
        selected_x = x[:, top_y_inds]  # shape: (k, input_dim)

        # Apply tanh to top-k scores and expand for broadcasting
        scaled_scores = torch.tanh(top_y_vals).unsqueeze(-1)  # shape: (k, 1)

        # Weight the selected x by their corresponding scores
        output = selected_x * scaled_scores  # shape: (k, input_dim
        
        return output 