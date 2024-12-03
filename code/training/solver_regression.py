"""Module containing the training logic for forecasting tasks on time-series."""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.linear_model import Ridge


class SolverRegression:
    
    def __init__(
        self, 
        device: torch.device,
        model: nn.Module, 
        train_data,
        val_data,
        test_data,
        classifier: str = 'svd',
        washout: int = 200,
        reg: float = 0,
    ) -> None:
        self.device = device
        self.model = model
        self.washout = washout

        if classifier == 'svd':
            self.classifier = Ridge(alpha=reg, solver='svd')
        else:
            raise ValueError(f"Invalid classifier: {classifier}. Only svd allowed")
        

        train_dataset, train_target = train_data
        val_dataset, val_target = val_data
        test_dataset, test_target = test_data
        if len(train_dataset.shape) == 1:
            train_dataset = train_dataset.reshape(1, -1, 1).to(device)
            train_target = train_target.reshape(-1, 1).numpy()
            val_dataset = val_dataset.reshape(1, -1, 1).to(device)
            val_target = val_target.reshape(-1, 1).numpy()
            test_dataset = test_dataset.reshape(1, -1, 1).to(device)
            test_target = test_target.reshape(-1, 1).numpy()
        
        self.train_dataset = train_dataset
        self.train_target = train_target
        self.val_dataset = val_dataset
        self.val_target = val_target
        self.test_dataset = test_dataset
        self.test_target = test_target

        self._reset()
    
    def _reset(self) -> None:
        self.results = {
            'train_nrmse': 0.0,
            'val_nrmse': 0.0,
            'test_nrmse': 0.0
        }

    @torch.no_grad()
    def evaluate(
        self, 
        dataset,
        target,
        scaler, 
    ) -> float:
        self.model.eval()
        
        outputs = self.model(dataset)[0].cpu().numpy()
        outputs = outputs[:, self.washout:]
        outputs = outputs.reshape(-1, self.model.n_units)
        outputs = scaler.transform(outputs)
        preds = self.classifier.predict(outputs)

        mse = np.mean(np.square(preds - target))
        rmse = np.sqrt(mse)
        norm = np.sqrt(np.square(target).mean())
        nrmse = rmse / (norm + 1e-9)
        return nrmse
    
    @torch.no_grad()
    def train(self) -> Tuple[float, float, float]:
        self.model.to(self.device)

        outputs = self.model(self.train_dataset)[0].cpu().numpy()
        outputs = outputs[:, self.washout:]
        outputs = outputs.reshape(-1, self.model.n_units)

        scaler = preprocessing.StandardScaler().fit(outputs)
        outputs = scaler.transform(outputs)
        self.classifier.fit(outputs, self.train_target)

        # Evaluate
        train_nrmse = self.evaluate(self.train_dataset, self.train_target, scaler)
        val_nrmse = self.evaluate(self.val_dataset, self.val_target, scaler)
        test_nrmse = self.evaluate(self.test_dataset, self.test_target, scaler)
        self.results = {
            'train_nrmse': train_nrmse,
            'val_nrmse': val_nrmse,
            'test_nrmse': test_nrmse
        }

        print(f"Train NRMSE: {train_nrmse:.4f}, Val NRMSE: {val_nrmse:.4f}, Test NRMSE: {test_nrmse:.4f}")

        return train_nrmse, val_nrmse, test_nrmse


        