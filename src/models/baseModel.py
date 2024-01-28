import numpy as np
import torch.optim as optim
from scipy.stats import loguniform, uniform
from sklearn.base import clone
from torch import nn


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


class LocalModel:
    def __init__(
        self,
        base_learner,
        encoder=None,
    ):
        self.encoder = encoder
        self.trained_classifiers = None
        self.base_learner = base_learner
        self.node_indicators = None
        self.path_eval = False

    def set_predictPath(self, val):
        self.path_eval = val

    def fit(self, X, y):
        self._fit_base_learner(X, y)
        return self

    def _fit_base_learner(self, X, y):
        encoded_y = self.encoder.transform(y)
        self.node_indicators = encoded_y.T
        self.trained_classifiers = np.zeros(encoded_y.shape[1], dtype=object)
        for idx, node_y in enumerate(self.node_indicators):
            unique_node_y = np.unique(node_y)
            if len(unique_node_y) == 1:
                cls = unique_node_y[0]
            else:
                cls = clone(self.base_learner)
                cls = cls.fit(X, node_y)
            self.trained_classifiers[idx] = cls

    def set_encoder(self, encoder):
        self.encoder = encoder

    def predict(self, X, threshold=0.5):
        pass

    def predict_proba(self, X):
        pass


tuning_space = {
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
    'module__neuron_power': range(8, 11),
    'module__dor_input': [0],
    'module__dor_hidden': uniform(0, 0.5)
}
