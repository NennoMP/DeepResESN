"""Module containing the training logic for memory-based tasks on time-series."""
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.linear_model import Ridge


class SolverMemCap:
    """
    TODO: Description something something.
    """
    
    def __init__(
        self, 
        device: torch.device,
        model: nn.Module, 
        washout: int = 100,
        classifier: str = 'svd',
        reg: float = 0,
    ):
        self.device = device
        self.model = model
        
        self.washout = washout

        if classifier == 'svd':
            self.classifier = Ridge(alpha=reg, solver='svd')
        else:
            raise ValueError(f"Invalid classifier: {classifier}. Only 'svd'")

    @torch.no_grad()
    def evaluate(
        self, 
        dataset,
        target,
        scaler, 
    ) -> Tuple[float, float]:
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(dataset)[0].cpu().numpy()
            outputs = outputs[:, self.washout:]
            outputs = outputs.reshape(-1, self.model.n_units)
            outputs = scaler.transform(outputs)
            preds = self.classifier.predict(outputs)
            
            target_mean = np.mean(target.cpu().numpy())
            pred_mean = np.mean(preds)
            num  = denom_t = denom_pred = 0
            for i in range(preds.shape[0]):
                deviat_t = target[i] - target_mean
                deviat_pred = preds[i] - pred_mean
                num += deviat_t * deviat_pred
                denom_t += deviat_t**2
                denom_pred += deviat_pred**2
            num = num**2
            den = denom_t * denom_pred
            
            return num / den
                
    @torch.no_grad()
    def train(self, nonlin: bool = False, verbose: bool = True):
        max_delay = self.model.n_units * 2

        self.model.to(self.device)
        
        train_MC = np.zeros(max_delay + 1)
        val_MC = np.zeros(max_delay + 1)
        test_MC = np.zeros(max_delay + 1)
        for k in range(1, max_delay + 1):
            
            # ############# generate random uniform time series data ############### #
            length = 7000
            time_series = torch.FloatTensor(length + k).uniform_(-0.8, 0.8)
            X_data = time_series[k:length+k]

            end_train = 5000
            trX = X_data[:end_train]
            train_dataset = trX[:end_train]
            
            end_val = end_train + 1000
            valX = X_data[end_train:]
            val_dataset = valX[:end_val]
            
            end_test = end_val + 1000
            tsX = X_data[end_val:]
            test_dataset = tsX[:end_test]

            k_delay = k
            if nonlin:
                train_target = torch.sin(torch.pi * time_series[self.washout:end_train])
                val_target = torch.sin(torch.pi * time_series[end_train+self.washout:-k_delay])
                test_target = torch.sin(torch.pi * time_series[end_val+self.washout:-k_delay])
            else:
                train_target = time_series[self.washout:end_train]
                val_target = time_series[end_train+self.washout:-k_delay]
                test_target = time_series[end_val+self.washout:-k_delay]

            train_dataset = train_dataset.reshape(1, train_dataset.shape[0], 1)
            val_dataset = val_dataset.reshape(1, val_dataset.shape[0], 1)
            test_dataset = test_dataset.reshape(1, test_dataset.shape[0], 1)
                
            # Fit training data
            with torch.no_grad():
                outputs = self.model(train_dataset)[0].cpu().numpy()
                outputs = outputs[:, self.washout:]
                outputs = outputs.reshape(-1, self.model.n_units)

            scaler = preprocessing.StandardScaler().fit(outputs)
            outputs = scaler.transform(outputs)
            self.classifier.fit(outputs, train_target)

            # Evaluate
            train_MC[k] = self.evaluate(train_dataset, train_target, scaler)
            val_MC[k] = self.evaluate(val_dataset, val_target, scaler)
            test_MC[k] = self.evaluate(test_dataset, test_target, scaler)

            if verbose:
                print(f'Delay k: {k}, Train MC: {train_MC[k]:.6f}, Val MC: {val_MC[k]:.6f}, , Test MC: {test_MC[k]:.6f}')
        
        train_MC = sum(train_MC)
        val_MC = sum(val_MC)
        test_MC = sum(test_MC)        
                
        return train_MC, val_MC, test_MC


        