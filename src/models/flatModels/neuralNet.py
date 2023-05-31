import torch
from torch import nn
import torch.nn.functional as F
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

import numpy as np

from ..wrapper import Wrapper
from skorch import NeuralNetClassifier

#TODO: configure Hiddden layers
class NeuralNet(nn.Module):
    def __init__(self, dim_in, dim_out, dropout_rate, num_units=1000, nonlin=nn.ReLU()):
        super().__init__()

        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout_rate)

        self.dense0 = nn.Linear(dim_in, num_units)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, dim_out)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.output(X)
        return X


params = dict(
        name='NeuralNet',
        model=NeuralNetClassifier(
            module=NeuralNet,
            max_epochs=25,
            criterion=nn.CrossEntropyLoss(),
            train_split=False, 
            verbose=0
        ),
        # preprocessing_steps=[('preprocessing', TruncatedSVD()),('StandardScaler', StandardScaler())],
        preprocessing_steps=[('StandardScaler', StandardScaler(with_mean=False))],
        # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        tuning_space={
                'lr': np.arange(0.001, 0.01, 0.005).tolist(),
                'batch_size': (16 * np.arange(1,8)).tolist(),
                'module__dropout_rate': [0.5]
        },
        data_shape_required=True    
)


# Please don't change this line
wrapper = Wrapper(**params)


if __name__ == "__main__":
    pass
