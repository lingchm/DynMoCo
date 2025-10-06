import torch
import torch.nn as nn
from typing import Tuple


class HGRUCell(nn.Module):
    def __init__(self, units: int, activation='tanh',
                    recurrent_activation='sigmoid',
                    use_bias=True,
                    kernel_initializer='xavier_uniform',
                    recurrent_initializer='orthogonal',
                    bias_initializer='zeros',
                    dtype=torch.float32):
        super(HGRUCell, self).__init__()

        self.units = int(units)
        self.activation = getattr(torch.nn.functional, activation) if hasattr(torch.nn.functional, activation) else torch.tanh
        self.recurrent_activation = getattr(torch.nn.functional, recurrent_activation) if hasattr(torch.nn.functional, recurrent_activation) else torch.sigmoid
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.dtype = dtype

    def build(self, input_shape):
        inp_shape = input_shape[0]
        rec_shape = input_shape[1]
        last_dim_inp = inp_shape[-1]
        last_dim_rec = rec_shape[-1]

        self.kernel_inp_x = nn.Parameter(torch.empty(last_dim_inp, 2*self.units, dtype=self.dtype))
        self.kernel_inp_h = nn.Parameter(torch.empty(last_dim_rec, 2*self.units, dtype=self.dtype))
        self.kernel_rec_x = nn.Parameter(torch.empty(last_dim_inp, self.units, dtype=self.dtype))
        self.kernel_rec_h = nn.Parameter(torch.empty(last_dim_rec, self.units, dtype=self.dtype))

        if self.use_bias:
            self.bias_inp = nn.Parameter(torch.empty(1, 2*self.units, dtype=self.dtype))
            self.bias_rec = nn.Parameter(torch.empty(1, self.units, dtype=self.dtype))
        else:
            self.bias_inp = None
            self.bias_rec = None

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        if self.kernel_initializer == 'xavier_uniform':
            nn.init.xavier_uniform_(self.kernel_inp_x)
            nn.init.xavier_uniform_(self.kernel_inp_h)
        elif self.kernel_initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.kernel_inp_x)
            nn.init.xavier_uniform_(self.kernel_inp_h)
        
        if self.recurrent_initializer == 'orthogonal':
            nn.init.orthogonal_(self.kernel_rec_x)
            nn.init.orthogonal_(self.kernel_rec_h)
        
        if self.use_bias:
            if self.bias_initializer == 'zeros':
                nn.init.zeros_(self.bias_inp)
                nn.init.zeros_(self.bias_rec)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        X, H = inputs
        
        # Build weights if not built yet
        if not hasattr(self, 'kernel_inp_x'):
            self.build([X.shape, H.shape])
        
        ZR = self.activation(torch.matmul(X, self.kernel_inp_x) + torch.matmul(H, self.kernel_inp_h))
        if self.bias_inp is not None:
            ZR = ZR + self.bias_inp
            
        Z, R = torch.split(ZR, self.units, dim=-1)
        H_new = self.recurrent_activation(torch.matmul(X, self.kernel_rec_x) + torch.matmul(R * H, self.kernel_rec_h))
        if self.bias_rec is not None:
            H_new = H_new + self.bias_rec
            
        H_new = (1 - Z) * H + Z * H_new
        return H_new

    def get_initial_state(self, input_shape) -> torch.Tensor:
        inp_shape = input_shape
        return torch.zeros(inp_shape[-1:] + [self.units], dtype=self.dtype)