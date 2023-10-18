import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import loguniform, uniform, randint


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, nonlin, num_hidden_layers, batch_norm,  dor_input, dor_hidden, neuron_power, en=None):
        super().__init__()

        MLP.en = en

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


tuning_space={
                'lr': loguniform(1e-5, 1e-3),
                # 'lr': [1e-5],
                # 'batch_size': (16 * np.arange(1,8)).tolist(),
                # 'batch_size': (16 * np.arange(1,4)).tolist(),
                'batch_size': [32, 64, 128],
                # 'optimizer': [optim.SGD, optim.Adam],
                'optimizer': [optim.Adam],
                'optimizer__weight_decay': loguniform(1e-6, 1e-4),
                # 'optimizer__momentum': loguniform(1e-3, 1e0),
                # 'module__nonlin': [nn.ReLU, nn.Tanh, nn.Sigmoid],
                'module__nonlin': [nn.ReLU],
                'module__batch_norm': [False],
                # 'module__num_hidden_layers': np.arange(0 , 8 , 2).tolist(),
                'module__num_hidden_layers': [1],
                # 'module__dor_input': uniform(0, 0.3),
                'module__neuron_power': range(9, 12),
                'module__dor_input': [0],
                'module__dor_hidden': uniform(0, 0.5)
}