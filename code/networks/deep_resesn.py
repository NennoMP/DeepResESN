"""Implementation of Deep Residual Echo State Network (DeepResESN) in PyTorch.

The structure of this file and the implementation of the DeepResESN class is heavily inspired by 
[1] and [2], where Deep Echo State Networks (DeepESN) and Residual Echo State Networks (ResESN) 
have been introduced, respectively.

References:
[1] https://github.com/gallicch/DeepRC-TF/blob/master/DeepRC.py
[2] https://github.com/andreaceni/ResESN
"""
from typing import Optional

import torch
import torch.nn as nn

import networks.reservoir as reservoir


class DeepResESN(nn.Module):
    def __init__(
        self,
        in_size: int = 1,
        concat: bool = False,
        n_layers: int = 1,
        n_units: int = 100,
        init: str = 'uniform',
        act: nn.Module = nn.Tanh(),
        config: reservoir.ReservoirConfig = reservoir.ReservoirConfig(),
        inter_config: Optional[reservoir.ReservoirConfig] = None,
    ) -> None:
        super().__init__()

        self.in_size = in_size
        self.concat = concat
        self.n_layers = n_layers
        self.n_units = n_units
        self.init = init
        self.act = act
        self.config = config
        if inter_config:
            self.inter_config = inter_config
        else:
            self.inter_config = config

        # if concat == True the number of units is evenly divided among the layers
        # Note: if an even distribution is not possible, the extra units are allocated to the 
        # first layer
        self.layers_units = self.first_layer_units = n_units 
        if concat:
            self.layers_units = n_units // n_layers
            self.first_layer_units = self.layers_units + n_units % n_layers
        
        self.layers = self._make_layers()

    def _make_layers(self) -> nn.Sequential:
        layers = []

        # First reservoir layer
        layers.append(
            reservoir.Reservoir(
                in_size=self.in_size, 
                n_units=self.first_layer_units, 
                init=self.init, 
                act=self.act, 
                config=self.config
            )
        )

        # Other reservoir layers
        # if concat == True the hidden size of the first layer may vary based on the remainder 
        # from the units distribution
        h_dim = self.first_layer_units
        for _ in range(1, self.n_layers):
            layers.append(
                reservoir.Reservoir(
                    in_size=h_dim, 
                    n_units=self.layers_units, 
                    init=self.init, 
                    act=self.act, 
                    config=self.inter_config
                )
            )
            h_dim = self.layers_units

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, N, T, H = self.n_layers, x.shape[0], x.shape[1], self.layers_units
        states = torch.empty(L, N, T, H, device=x.device)
        last_states = torch.empty(L, N, H, device=x.device)

        
        for i, layer in enumerate(self.layers):
            x, last_h = layer(x)
            states[i, :, :, :] = x
            last_states[i, :, :] = last_h
        
        if self.concat:
            states = states.view(N, T, -1)
            last_states = last_states.view(N, -1)
        else:
            states = states[-1, :, :, :]
            last_states = last_states[-1, :, :]
        
        return states, last_states
    
    
def get_deepresesn(hparams: dict) -> DeepResESN:
    cls = DeepResESN
    init_params = cls.__init__.__code__.co_varnames[1:cls.__init__.__code__.co_argcount]
    values = cls.__init__.__defaults__
    params_dict = dict(zip(init_params[-len(values):], values))
    
    params = {**params_dict, **hparams}
    # remove params not present
    params = {k: v for k, v in params.items() if k in init_params}

    return cls(**params)