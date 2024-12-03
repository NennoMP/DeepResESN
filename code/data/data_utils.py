"""Module containing utils for data loading and pre-processing. Code heavily adapted from [1].

References:
[1] https://github.com/andreaceni/ResESN/blob/main/utils.py
"""
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from scipy.integrate import odeint
from scipy.io import arff
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


##############################
# MEMORY-BASED DATASETS
##############################
def get_ctXOR(
    data_dir: str, 
    delay: int = 5, 
    lag: int = 1, 
    washout: int = 200
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Get the continuous-time XOR dataset (ctXOR).
    
    Args:
        data_dir: the directory where the dataset should be stored/loaded.
        delay: the delay of the XOR function.
        lag: the lag of the time series.
        washout: the washout period.

    Raises:
        ValueError if the delay is invalid.

    Returns:
        A tuple of train, validation, and test datasets.
    """
    if delay in [5, 10]:
        dataset = pd.read_csv(f'{data_dir}/memory/ctXOR/ctxor{delay}.csv', header=None).transpose()
    else:
        raise ValueError(f"Invalid ctXOR delay: {delay}. Options are '5' and '10'!")
    
    # 6k steps
    x = torch.tensor(dataset.iloc[0].values).float()
    y = torch.tensor(dataset.iloc[1].values).float()
    
    # 4k train - 1k val - 1k test
    end_train = 4000
    end_val = end_train + 1000
    end_test = end_val + 1000
    
    train_dataset = x[:end_train-lag]
    train_target = y[washout+lag:end_train]

    val_dataset = x[end_train:end_val-lag]
    val_target = y[end_train+washout+lag:end_val]

    test_dataset = x[end_val:end_test-lag]
    test_target = y[end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)

def get_sinmem(
    data_dir: str, 
    delay: int = 10, 
    lag: int = 1, 
    washout: int = 200
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Get the SinMem dataset.
    
    Args:
        data_dir: the directory where the dataset should be stored/loaded.
        delay: the delay of the SinMem function.
        lag: the lag of the time series.
        washout: the washout period.

    Raises:
        ValueError if the delay is invalid.

    Returns:
        A tuple of train, validation, and test datasets.
    """
    if delay in [10, 20]:
        dataset = pd.read_csv(f'{data_dir}/memory/SinMem/sinmem{delay}.csv', header=None).transpose()
    else:
        raise ValueError(f"Invalid SinMem delay: {delay}. Options are '10' and '20'!")
    
    # 6k steps
    x = torch.tensor(dataset.iloc[0].values).float()
    y = torch.tensor(dataset.iloc[1].values).float()
    
    # 4k train - 1k val - 1k test
    end_train = 4000
    end_val = end_train + 1000
    end_test = end_val + 1000
    
    train_dataset = x[:end_train-lag]
    train_target = y[washout+lag:end_train]

    val_dataset = x[end_train:end_val-lag]
    val_target = y[end_train+washout+lag:end_val]

    test_dataset = x[end_val:end_test-lag]
    test_target = y[end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)

##############################
# FORECASTING DATASETS
##############################
def get_narma(
    data_dir: str, 
    delay: int, 
    lag: int = 1, 
    washout: int = 200
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Get the NARMA dataset.
    
    Args:
        data_dir: the directory where the dataset should be stored/loaded.
        delay: the delay of the SinMem function.
        lag: the lag of the time series.
        washout: the washout period.

    Raises:
        ValueError if the delay is invalid.

    Returns:
        A tuple of train, validation, and test datasets.
    """
    if delay in [30, 60]:
        dataset = pd.read_csv(
            f'{data_dir}/forecasting/NARMA/narma{delay}.csv', header=None
        ).transpose()
    else:
        raise ValueError(f"Invalid NARMA delay: {delay}. Options are '30' and '60'!")
        
    # 10k steps
    x = torch.tensor(dataset.iloc[0].values).float()
    y = torch.tensor(dataset.iloc[1].values).float()
    
    # 5k train - 2.5k val - 2.5k test
    end_train = 5000
    end_val = end_train + 2500
    end_test = end_val + 2500
    
    train_dataset = x[:end_train-lag]
    train_target = y[washout+lag:end_train]

    val_dataset = x[end_train:end_val-lag]
    val_target = y[end_train+washout+lag:end_val]

    test_dataset = x[end_val:end_test-lag]
    test_target = y[end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)

def create_lorenz(
    N: int = 5, 
    F: int = 8, 
    num_batch: int = 128, 
    lag: int = 25, 
    washout: int = 200, 
    window_size: int = 0, 
    serieslen: int = 2
) -> torch.Tensor:
    """Create a Lorenz dataset.

    Args:  
        N: number of variables
        F: forcing term
        num_batch: number of batches
        lag: lag
        washout: washout
        window_size: window size
        serieslen: series length

    Returns:
        A Lorenz time series.
    """
    def L96(x, t: float) -> np.ndarray:
        """
        Lorenz 96 model with constant forcing
        # https://en.wikipedia.org/wiki/Lorenz_96_model
        """
        # Setting up vector
        d = np.zeros(N)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(N):
            d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return d
    
    def get_fixed_length_windows(
        tensor, 
        length: int, 
        prediction_lag: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(tensor.shape) <= 2
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(-1)

        windows = tensor[:-prediction_lag].unfold(0, length, 1)
        windows = windows.permute(0, 2, 1)

        targets = tensor[length+prediction_lag-1:]
        return windows, targets  # input (B, L, I), target, (B, I)

    dt = 0.01
    t = np.arange(0.0, serieslen+(lag*dt)+(washout*dt), dt)
    dataset = []
    for _ in range(num_batch):
        x0 = np.random.rand(N) + F - 0.5 # [F-0.5, F+0.5]
        x = odeint(L96, x0, t)
        dataset.append(x)
    dataset = np.stack(dataset, axis=0)
    dataset = torch.from_numpy(dataset).float()

    if window_size > 0:
        windows, targets = [], []
        for i in range(dataset.shape[0]):
            w, t = get_fixed_length_windows(dataset[i], window_size, prediction_lag=lag)
        windows.append(w)
        targets.append(t)
        return torch.utils.data.TensorDataset(torch.cat(windows, dim=0), torch.cat(targets, dim=0))
    else:
        return dataset

def get_lorenz(
    N: int = 5, 
    F: int = 8, 
    num_batch: int = 128, 
    lag: int = 25, 
    washout: int = 200, 
    window_size: int = 0, 
    serieslen: int = 2
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Get the Lorenz dataset.
    
    Args:
        N: number of variables
        F: forcing term
        num_batch: number of batches
        lag: lag
        washout: washout
        window_size: window size
        serieslen: series length

    Returns:  
        A tuple of train, validation, and test datasets. 
    """
    train_dataset = create_lorenz(N=N, F=F, num_batch=num_batch, lag=lag, washout=washout, window_size=window_size, serieslen=serieslen)
    val_dataset = create_lorenz(N=N, F=F, num_batch=num_batch, lag=lag, washout=washout, window_size=window_size, serieslen=serieslen)
    test_dataset = create_lorenz(N=N, F=F, num_batch=num_batch, lag=lag, washout=washout, window_size=window_size, serieslen=serieslen)
    
    train_target = torch.tensor(train_dataset[:, (lag+washout):].numpy().reshape(-1, N))
    train_dataset = train_dataset[:, :(100*serieslen+washout)]
    
    val_target = torch.tensor(val_dataset[:, (lag+washout):].numpy().reshape(-1, N))
    val_dataset = val_dataset[:, :(100*serieslen+washout)]
    
    test_target = torch.tensor(test_dataset[:, (lag+washout):].numpy().reshape(-1, N))
    test_dataset = test_dataset[:, :(100*serieslen+washout)]
    
    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)

def get_mackey_glass(
    data_dir: str, 
    lag: int = 1, 
    washout: int = 200, 
    window_size: int = 0
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Get the Mackey-Glass dataset.
    
    Args: 
        data_dir: the directory where the dataset should be stored/loaded.
        lag: the lag of the time series.
        washout: the washout period.
        window_size: the window size.

    Returns:
        A tuple of train, validation, and test datasets.
    """
    with open(f'{data_dir}/forecasting/MG/mackey-glass.csv', 'r') as f:
        dataset = f.readlines()[0]  # single line file

    # 10k steps
    dataset = torch.tensor([float(el) for el in dataset.split(',')]).float()

    if window_size > 0:
        assert washout == 0
        dataset, targets = get_fixed_length_windows(dataset, window_size, prediction_lag=lag)
        # dataset is input, targets is output

        end_train = int(dataset.shape[0] / 2)
        end_val = end_train + int(dataset.shape[0] / 4)
        end_test = dataset.shape[0]

        train_dataset = dataset[:end_train]
        train_target = targets[:end_train]

        val_dataset = dataset[end_train:end_val]
        val_target = targets[end_train:end_val]

        test_dataset = dataset[end_val:end_test]
        test_target = targets[end_val:end_test]
    else:
        end_train = int(dataset.shape[0] / 2)
        end_val = end_train + int(dataset.shape[0] / 4)
        end_test = dataset.shape[0]

        train_dataset = dataset[:end_train-lag]
        train_target = dataset[washout+lag:end_train]

        val_dataset = dataset[end_train:end_val-lag]
        val_target = dataset[end_train+washout+lag:end_val]

        test_dataset = dataset[end_val:end_test-lag]
        test_target = dataset[end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)


##############################
# CLASSIFICATION DATASETS
##############################
def get_classification_dataset(
    data_dir: str, 
    dataset_name: str,
    batch_size: int = 128, 
    val_split: float = 0.3,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load a specified classification dataset.
    
    Args:
        data_dir: the directory where the dataset should be stored/loaded
        dataset_name: the name of the dataset to load

    Returns:
        A tuple of train, validation, and test dataloaders
    """
    if dataset_name == 'Kepler':
        train_dataset = pd.read_csv(f'{data_dir}/classification/{dataset_name}/train.ts', skiprows=7, header=None)
        test_dataset = pd.read_csv(f'{data_dir}/classification/{dataset_name}/test.ts', skiprows=7, header=None)
    else:
        train_dataset = pd.DataFrame(arff.loadarff(f'{data_dir}/classification/{dataset_name}/train.arff')[0])
        test_dataset = pd.DataFrame(arff.loadarff(f'{data_dir}/classification/{dataset_name}/test.arff')[0])
    
    full_train_target = torch.tensor([int(el) for el in train_dataset.iloc[:, -1].values]).int()
    full_train_dataset = torch.tensor(train_dataset.iloc[:, :-1].values).float()
    test_target = torch.tensor([int(el) for el in test_dataset.iloc[:, -1].values]).int()
    test_dataset = torch.tensor(test_dataset.iloc[:, :-1].values).float()
    
    # Train-Val stratified split
    train_dataset, val_dataset, train_target, val_target = train_test_split(
        full_train_dataset,
        full_train_target,
        test_size=val_split,
        stratify=full_train_target,
        random_state=random_state,
    )

    # Dataloaders
    full_train_dataset = TensorDataset(full_train_dataset, full_train_target)
    train_dataset = TensorDataset(train_dataset, train_target)
    val_dataset = TensorDataset(val_dataset, val_target)
    test_dataset = TensorDataset(test_dataset, test_target)
    
    
    full_train_dataloader = DataLoader(
        full_train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=1,
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=1,
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=1,
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=1,
    )
    
    return full_train_dataloader, train_dataloader, val_dataloader, test_dataloader

def get_mnist(
    data_dir: str, 
    batch_size: int = 128, 
    normalize: bool = False,
    val_split: float = 0.05,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get the MNIST dataset.

    Args:
        data_dir: the directory where the dataset should be stored/loaded
        batch_size: the batch size
        normalize: whether to normalize the dataset
        val_split: the ratio of validation split

    Returns:
        A tuple of train, validation, and test dataloaders
    """
    MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
    
    # Transformations
    normalization = None
    if normalize:
        normalization = torchvision.transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)

    transform = [torchvision.transforms.ToTensor()]
    if normalization:
        transform.append(normalization)
    transform = torchvision.transforms.Compose(transform)

    dataset = torchvision.datasets.MNIST
    full_train_dataset = dataset(data_dir, train=True, transform=transform, download=True)
    test_dataset = dataset(data_dir, train=False, transform=transform, download=True)
    
    # Train-Val split
    val_size = np.int64(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, 
        [train_size, val_size]
    )
    
    # Dataloaders
    full_train_dataloader = DataLoader(
        full_train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=1,
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=1,
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=1,
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=1,
    )

    return full_train_dataloader, train_dataloader, val_dataloader, test_dataloader