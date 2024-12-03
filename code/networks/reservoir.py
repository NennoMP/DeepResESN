"""Implementation of recurrent layers and recurrent cells of DeepResESN, in PyTorch.

The structure of this file and the implementation of the DeepResESN class is heavily inspired by 
[1] and [2], where Deep Echo State Networks (DeepESN) and Residual Echo State Networks (ResESN) 
have been introduced, respectively.

References:
[1] https://github.com/gallicch/DeepRC-TF/blob/master/DeepRC.py
[2] https://github.com/andreaceni/ResESN
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

import networks.init_utils as init_utils


@dataclass
class ReservoirConfig:
    """DataClass containing the hyper-parameter for a layer of DeepResESN."""
    rho: float = 0.9
    alpha: float = 1
    beta: float = 1
    in_scaling: float = 1
    bias_scaling: float = 0
    # 1: fully-connected, otherwise the % of connections (e.g. 0.5 for 50%, etc.)
    connectivity: float = 1


class ReservoirCell(nn.Module):
    def __init__(
        self, 
        in_size: int, 
        n_units: int, 
        init: str,
        act: nn.Module,
        config: ReservoirConfig,
    ) -> None:  
        super().__init__()    
        
        self.n_units = n_units
        self.act = act
        self.alpha = nn.Parameter(torch.tensor([config.alpha]), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor([config.beta]), requires_grad=False)
        
        connectivity = n_units * config.connectivity
        
        self.in_kernel = init_utils.sparse_tensor_init(
            M=in_size, 
            N=n_units,
            C=connectivity
        ) * config.in_scaling
        
        self.recurrent_kernel = init_utils.init_recurrent_kernel(
            n_units=n_units,
            connectivity=connectivity,
            rho=config.rho,
            init=init
        )
        
        self.bias = init_utils.init_bias(n_units=n_units, bias_scaling=config.bias_scaling)
        
        # Random orthogonal matrix
        Q, _ = torch.linalg.qr(2 * torch.rand(n_units, n_units) - 1)
        self.ortho = nn.Parameter(Q, requires_grad=False)
    
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Computes the output of the cell given the input and previous state."""
        input_part = torch.mm(x, self.in_kernel)
        state_part = torch.mm(h_prev, self.recurrent_kernel)
        out = self.act(state_part + input_part + self.bias)

        h_prev = torch.mm(h_prev, self.ortho) # Apply linear orthogonal transformation
        out = (h_prev * self.alpha) + (out * self.beta)
    
        return out


class Reservoir(nn.Module):
    def __init__(
        self, 
        in_size: int,
        n_units: int,  
        init: str,
        act: nn.Module,
        config: ReservoirConfig,
    ) -> None:
        super().__init__()

        self.net = ReservoirCell(
            in_size=in_size, 
            n_units=n_units, 
            init=init, 
            act=act,
            config=config,
        )

    def _init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.net.n_units)

    def forward(
        self, 
        x: torch.Tensor, 
        h_prev: Optional[torch.Tensor] = None, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, T = x.shape[0], x.shape[1]
        states = torch.empty(batch_size, T, self.net.n_units, device=x.device)

        if h_prev is None:
            h_prev = self._init_hidden(batch_size).to(x.device)
            
        for t in range(T):
            h_prev = self.net(x[:, t], h_prev)
            states[:, t] = h_prev

        return states, states[:, -1, :]