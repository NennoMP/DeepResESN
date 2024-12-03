"""Module containing the training logic for classification tasks on time-series."""
from typing import List, Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.linear_model import RidgeClassifier
from torch.utils.data import DataLoader


class SolverClassification:
    
    def __init__(
        self, 
        device: torch.device,
        model: nn.Module, 
        train_dataloader: DataLoader = None, 
        val_dataloader: DataLoader = None, 
        test_dataloader: DataLoader = None, 
        permutation: Optional[List[int]] = None,
        classifier: str = 'svd',
        reg: float = 0,
    ):
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        if classifier == 'svd':
            self.classifier = RidgeClassifier(alpha=reg, solver='svd')
        else:
            raise ValueError(f"Invalid classifier: {classifier}. Only svd allowed.")
        
        self.seqlen = self._get_seqlen(train_dataloader)
            
        # for psMNIST
        self.permutation = None
        if permutation:
            self.permutation = torch.Tensor(permutation).to(torch.long).to(device)
        
        self._reset()
    
    def _get_seqlen(self, dataloader: DataLoader) -> int:
        first_batch = next(iter(dataloader))

        MNIST_SEQLEN = 784
        if len(first_batch[0].shape) == 4:
            return MNIST_SEQLEN
        return first_batch.shape[-1]

    def _reset(self) -> None:
        self.results = {
            'train_acc': 0.0,
            'val_acc': 0.0,
            'test_acc': 0.0
        }

    @torch.no_grad()
    def evaluate(
        self, 
        dataloader: DataLoader, 
        scaler, 
    ) -> float:
        self.model.eval()
        
        labels, outputs = [], []
        for x, y in tqdm(dataloader):
            x = x.to(self.device)
            x = x.reshape(x.shape[0], 1, self.seqlen) 
            x = x.permute(0, 2, 1)
            if self.permutation:
                x = x[:, self.permutation, :]
            out = self.model(x)[1]
            labels.append(y)
            outputs.append(out.cpu())

        labels = torch.cat(labels, dim=0).numpy()
        outputs = torch.cat(outputs, dim=0).numpy()
        preds = scaler.transform(outputs)

        return self.classifier.score(preds, labels)
    
    @torch.no_grad()
    def train(self) -> Tuple[float, float, float]:
        self.model.to(self.device)

        labels, outputs = [], []
        for x, y in tqdm(self.train_dataloader):
            x = x.to(self.device)
            x = x.reshape(x.shape[0], 1, self.seqlen)
            x = x.permute(0, 2, 1)
            if self.permutation:
                x = x[:, self.permutation, :]
            out = self.model(x)[1]
            labels.append(y)
            outputs.append(out.cpu())

        labels = torch.cat(labels, dim=0).numpy()
        outputs = torch.cat(outputs, dim=0).numpy()
        scaler = preprocessing.StandardScaler().fit(outputs)
        outputs = scaler.transform(outputs)
        self.classifier.fit(outputs, labels)

        # Evaluate
        train_acc = self.classifier.score(outputs, labels)
        val_acc = self.evaluate(self.val_dataloader, scaler)
        test_acc = self.evaluate(self.test_dataloader, scaler)
        self.results = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc
        }

        print(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")

        return train_acc, val_acc, test_acc


        