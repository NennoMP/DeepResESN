"""Module containing utility functions for initialization. Code heavily adapted from [1] and [2].

References:
[1] https://github.com/gallicch/DeepRC-TF/blob/master/DeepRC.py
[2] https://github.com/andreaceni/ResESN
"""
import numpy as np
import torch
import torch.nn as nn


#########################
# INIT RESIDUAL CONNECTIONS UTILS
#########################
def init_shortcut(
    device: torch.device, 
    M: int,
    N: int,
    delta: float = 0.1,
    skip_option: str = 'ortho', 
) -> None:
    """
    Utility to initialize the residual connections of DeepResESN.
    """

    # Identity
    if skip_option == 'identity':
        O = np.eye(M, N)
    # Random orthogonal
    elif skip_option == 'ortho':
        O, _ = np.linalg.qr(2 * np.random.rand(M, N) - 1)
    # Cycle orthogonal
    elif skip_option == 'cycle':
        O = np.zeros((M, N))
        O[0, N - 1] = 1
        O[torch.arange(1, M), torch.arange(N - 1)] = 1
    # Skew orthogonal
    elif skip_option == 'skew_ortho':
        A = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float)
        # repeat matrix A to fill matrix (n_units x n_units)
        O = torch.block_diag(*[A for _ in range(N // 2)])
    # Entangled
    elif skip_option == 'entangled':
        ones = np.ones((M, N))
        I = np.eye(M, N)
        O = (1 / M * ones * delta) + ((1 - delta) * I)
    else:
        raise ValueError(
            f"Invalid skip option: {skip_option}. Only identity, ortho, cycle, skew_ortho or entangled allowed")
        
    return nn.Parameter(torch.Tensor(O).to(device), requires_grad=False)


#########################
# INIT TENSOR UTILS
#########################
def sparse_tensor_init(M: int, N: int, C: int = 1) -> torch.FloatTensor:
    """ Generates an M x N matrix to be used as sparse (input) kernel
    For each row only C elements are non-zero (i.e., each input dimension is
    projected only to C neurons). The non-zero elements are generated randomly
    from a uniform distribution in [-1,1]

    :param M: number of rows
    :param N: number of columns
    :param C: number of nonzero elements
    :return: MxN dense matrix
    """
    dense_shape = torch.Size([M, N])  # shape of the dense version of the matrix
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    k = 0
    for i in range(M):
        # the indices of non-zero elements in the i-th row of the matrix
        idx = np.random.choice(N, size=C, replace=False)
        for j in range(C):
            indices[k, 0] = i
            indices[k, 1] = idx[j]
            k = k + 1
    values = 2 * np.random.rand(M * C).astype('f') - 1
    values = torch.from_numpy(values)
    return nn.Parameter(
        torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float(), requires_grad=False
    )

def init_recurrent_kernel(
    n_units: int,
    connectivity: int,
    rho: float,
    init: str,
) -> nn.Parameter:
    """Utility method to initialize the recurrent kernel of the ResESN (non-Euler)."""
    W = sparse_recurrent_tensor_init(n_units, C=connectivity, init=init)

    if init == 'normal':
        W = rho * W # NB: W was already rescaled to 1 (circular law)
    elif init == 'uniform' and connectivity == n_units: # fully connected uniform
        W = fast_spectral_rescaling(W, rho)
    else: # sparse connections uniform
        W = spectral_norm_scaling(W, rho)
    recurrent_kernel = W
        
    return nn.Parameter(recurrent_kernel, requires_grad=False)

def sparse_recurrent_tensor_init(M: int, C: int, init: str = 'uniform') -> torch.Tensor:
    """ Generates an M x M matrix to be used as sparse recurrent kernel.
    For each column only C elements are non-zero (i.e., each recurrent neuron
    take sinput from C other recurrent neurons). The non-zero elements are
    generated randomly from a uniform distribution in [-1,1].

    :param M: number of hidden units
    :param C: number of nonzero elements
    :param distrib: initialisation strategy. It can be 'uniform' or 'normal'
    :return: MxM dense matrix
    """
    assert M >= C
    dense_shape = torch.Size([M, M])  # the shape of the dense version of the matrix
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    k = 0
    for i in range(M):
        # the indices of non-zero elements in the i-th column of the matrix
        idx = np.random.choice(M, size=C, replace=False)
        for j in range(C):
            indices[k, 0] = idx[j]
            indices[k, 1] = i
            k = k + 1
    if init == 'uniform':
        values = 2 * np.random.rand(M * C).astype('f') - 1
    elif init == 'normal':
        values =  np.random.randn(M * C).astype('f') / np.sqrt(C) # circular law (rescaling)
    else:
        raise ValueError(f"Invalid initialization {init}. Only uniform and normal allowed.")
    values = torch.from_numpy(values)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()

def init_bias(
    n_units: int,  
    bias_scaling: float, 
) -> nn.Parameter:
    """Utility method to initialize the bias according to a scaling parameter."""
    bias = torch.empty(n_units).uniform_(-bias_scaling, bias_scaling)
    return nn.Parameter(bias, requires_grad=False)


#########################
# SPECTRAL RESCALING UTILS
#########################
def spectral_norm_scaling(W: torch.Tensor, rho_desired: float) -> torch.Tensor:
    """ Rescales W to have rho(W) = rho_desired

    :param W:
    :param rho_desired:
    :return:
    """
    e, _ = np.linalg.eig(W.cpu())
    rho_curr = max(abs(e))
    return W * (rho_desired / rho_curr)

def fast_spectral_rescaling(W: torch.Tensor, rho_desired: float) -> torch.Tensor:
    """Rescales a W uniformly sampled in (-1,1) to have rho(W) = rho_desired. 
    This method is fast since we don't need to compute the spectrum of W, which is very slow.

    NB: this method works only if W is uniformly sampled in (-1,1). 
    In particular, W must be fully connected!

    :param W: must be a square matrix uniformly sampled in (-1,1), fully connected.
    :param rho_desired:
    :return:
    """
    units = W.shape[0]
    value  = (rho_desired / np.sqrt(units)) * (6 / np.sqrt(12))
    return W * value