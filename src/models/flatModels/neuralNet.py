import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import cfg
import pandas as pd
import numpy as np
from math import ceil

from models.wrapper import WrapperNN
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from scipy.stats import loguniform, uniform, randint
from utilities.customizedValidSplit import CustomizedValidSplit
from calibration.calibrate_model import CalibratedClassifier


class NeuralNet(nn.Module):
    def __init__(self, dim_in, dim_out, nonlin, num_hidden_layers, batch_norm,  dor_input, dor_hidden, neuron_power, **kwargs):
        super().__init__()

        layers = []
        fixed_neuron_num = 2 ** neuron_power
        
        # Configure input layer
        layer = [
            nn.Linear(dim_in, fixed_neuron_num), 
            # nn.Dropout(dor_input), 
            nonlin()
        ]
        if batch_norm:
            layer.append(nn.BatchNorm1d(fixed_neuron_num))

        layers.extend(layer)

        # Configure hidden layers
        for i in range(num_hidden_layers):
            layer = [
                nn.Linear(fixed_neuron_num, fixed_neuron_num), 
                nonlin()
            ]
            if batch_norm:
                layer.append(nn.BatchNorm1d(fixed_neuron_num))

            layer.append(nn.Dropout(dor_hidden))
            layers.extend(layer)

        # Configure output layer
        layers.append(nn.Linear(fixed_neuron_num, dim_out))
        self.layers_seq = nn.Sequential(*layers)

    def forward(self, X, **kwargs):
        X = self.layers_seq(X)
        return X


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

tuning_space={
                'lr': loguniform(1e-3, 1e-1),
                # 'batch_size': (16 * np.arange(1,8)).tolist(),
                # 'batch_size': (16 * np.arange(1,4)).tolist(),
                'batch_size': [4, 16, 32],
                # 'optimizer': [optim.SGD, optim.Adam],
                'optimizer': [optim.Adam],
                'optimizer__weight_decay': loguniform(1e-4, 3e-4),
                # 'optimizer__momentum': loguniform(1e-3, 1e0),
                # 'module__nonlin': [nn.ReLU, nn.Tanh, nn.Sigmoid],
                'module__nonlin': [nn.ReLU],
                'module__batch_norm': [True, False],
                # 'module__num_hidden_layers': np.arange(0 , 8 , 2).tolist(),
                'module__num_hidden_layers': [1],
                # 'module__dor_input': uniform(0, 0.3),
                'module__neuron_power': range(9, 12),
                'module__dor_input': [0],
                'module__dor_hidden': uniform(0, 0.5)
}

params = dict(
        name='NeuralNet',
        model=NeuralNetClassifier(
            module=NeuralNet,
            max_epochs=1 if cfg.debug else 30,
            criterion=nn.CrossEntropyLoss(),
            # train_split=CustomizedValidSplit(cv=0.15, stratified=True, random_state=None), # set later In case of intraDataset 
            # train_split=ValidSplit(cv=0.15, stratified=True, random_state=None), # set later In case of intraDataset 
            train_split=None,
            verbose=0,
            # callbacks=[EarlyStopping(patience=5)], 
            device=device
        ),
        # preprocessing_steps=[('preprocessing', TruncatedSVD()),('StandardScaler', StandardScaler())],
        # preprocessing_steps=[('StandardScaler', StandardScaler(with_mean=False))],
        preprocessing_steps=[('StandardScaler', StandardScaler())],
        # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        tuning_space=tuning_space,
        calibrater=CalibratedClassifier(criterion=nn.CrossEntropyLoss(), method='TS', lr=0.001)
)


# Please don't change this line
wrapper = WrapperNN(**params)


if __name__ == "__main__":
    pass
