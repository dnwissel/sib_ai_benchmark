import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

import numpy as np

from ..wrapper import Wrapper
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from scipy.stats import loguniform, uniform, randint

#TODO: configure Hiddden layers
num_hidden_layers = 5
kwargs = {}

# for i in range(num_hidden_layers):
#     kwargs[f'dr_l{i}'] = 0.5
#     kwargs[f'neuron_l{i}'] = 0.5
# kwargs[f'neuron_l{num_hidden_layers}'] = 0.1

# def test(**kwargs):
#     print(kwargs['dr_l0'])

class NeuralNet(nn.Module):
    def __init__(self, dim_in, dim_out, nonlin, num_hidden_layers,  dor_input, dor_hidden, neuron_ratio=0.5, **kwargs): # here is definition of the func
        super().__init__()

        layers = []
        fixed_neuron_num = round(neuron_ratio * dim_in // 16)
        # Configure input layer
        layers.extend([
            nn.Linear(dim_in, fixed_neuron_num), 
            nn.Dropout(dor_input), 
            nonlin()
        ])

        # Configure hidden layers
        for i in range(num_hidden_layers):
            layers.extend([
                nn.Linear(fixed_neuron_num, fixed_neuron_num), 
                nn.Dropout(dor_hidden), 
                nonlin()
            ])

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
# print(device)

for i in range(num_hidden_layers):
    kwargs[f'module__dr_l{i}'] = loguniform(1e-2, 1e0)
    kwargs[f'module__neuron_l{i}'] = [1, 2, 0.1, 0.01]
kwargs[f'module__neuron_l{num_hidden_layers}'] = [1, 2, 0.1, 0.01]

tuning_space={
                'lr': loguniform(1e-3, 1e2),
                'batch_size': (16 * np.arange(1,8)).tolist(),
                'optimizer': [optim.SGD, optim.Adam],
                # 'optimizer__momentum': loguniform(1e-3, 1e0),
                'module__nonlin': [nn.ReLU, nn.Tanh, nn.Sigmoid],
                # 'module__num_hidden_layers': np.arange(0 , 8 , 2).tolist(),
                'module__num_hidden_layers': [3],
                # 'module__dor_input': uniform(0, 0.3),
                'module__dor_input': [0],
                'module__dor_hidden': uniform(0, 1)
}

# tuning_space.update(kwargs)

params = dict(
        name='NeuralNet',
        model=NeuralNetClassifier(
            module=NeuralNet,
            max_epochs=25,
            criterion=nn.CrossEntropyLoss(),
            train_split=ValidSplit(5), # set later In case of intraDataset 
            verbose=0,
            callbacks=[EarlyStopping(patience=5)], # use external validation dataset from gridsearch?
            device=device
        ),
        # preprocessing_steps=[('preprocessing', TruncatedSVD()),('StandardScaler', StandardScaler())],
        preprocessing_steps=[('StandardScaler', StandardScaler(with_mean=False))],
        # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        tuning_space=tuning_space,
        data_shape_required=True    
)


# Please don't change this line
wrapper = Wrapper(**params)


if __name__ == "__main__":
    pass
