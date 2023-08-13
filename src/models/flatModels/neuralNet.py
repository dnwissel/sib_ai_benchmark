import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np

from ..wrapper import Wrapper
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from scipy.stats import loguniform, uniform, randint



class WrapperNN(Wrapper):
        def __init__(self, model, name, tuning_space=None, preprocessing_steps=None, preprocessing_params=None, is_selected=True): 
                super().__init__(model, name, tuning_space, preprocessing_steps, preprocessing_params, is_selected)

        def init_model(self, X, train_y_label, test_y_label):
            # num_feature, num_class = X.shape[1], train_y_label.nunique()
            print(X.shape[1], train_y_label.nunique())
            num_feature, num_class = X.shape[1], train_y_label.nunique()
            # num_feature, num_class = splits[0][0][0].shape[1], len(set(splits[0][0][1].nunique()) | set(splits[0][1][1].nunique()))
            self.model.set_params(module__dim_in=num_feature, module__dim_out=num_class) #TODO num_class is dependent on training set

            # Define pipeline and param_grid
            param_grid = {}
            if self.tuning_space:
                for key, value in self.tuning_space.items():
                    param_grid[self.name + '__' + key] = value

            if not self.preprocessing_steps:
                pipeline = Pipeline([(self.name, self.model)])
            else:
                pipeline = Pipeline(self.preprocessing_steps + [(self.name, self.model)])
                
                if self.preprocessing_params:
                    param_grid.update(self.preprocessing_params)
            return pipeline, param_grid, train_y_label, test_y_label

        def predict_proba(self, model_fitted, X):
            proba = model_fitted.predict_proba(X)
            pl_0 = Pipeline(model_fitted.best_estimator_.steps[:-1])
            net = model_fitted.best_estimator_.steps[-1][1] #TODO: make a copy
            # return proba, F.softmax(net.forward(pl_0.transform(X)), dim=-1)
            return proba, net.forward(pl_0.transform(X))


class NeuralNet(nn.Module):
    def __init__(self, dim_in, dim_out, nonlin, num_hidden_layers,  dor_input, dor_hidden, neuron_power, **kwargs):
        super().__init__()

        layers = []
        # fixed_neuron_num = round(neuron_power * dim_in) - round(neuron_power * dim_in) % 16
        fixed_neuron_num = 2 ** neuron_power
        # Configure input layer
        layers.extend([
            nn.Linear(dim_in, fixed_neuron_num), 
            # nn.Dropout(dor_input), 
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

tuning_space={
                'lr': loguniform(1e-3, 1e0),
                'batch_size': (16 * np.arange(1,8)).tolist(),
                # 'optimizer': [optim.SGD, optim.Adam],
                'optimizer': [optim.Adam],
                # 'optimizer__momentum': loguniform(1e-3, 1e0),
                'module__nonlin': [nn.ReLU, nn.Tanh, nn.Sigmoid],
                # 'module__num_hidden_layers': np.arange(0 , 8 , 2).tolist(),
                'module__num_hidden_layers': [1],
                # 'module__dor_input': uniform(0, 0.3),
                'module__neuron_power': range(8, 13),
                'module__dor_input': [0],
                'module__dor_hidden': uniform(0, 1)
}

params = dict(
        name='NeuralNet',
        model=NeuralNetClassifier(
            module=NeuralNet,
            # max_epochs=30,
            max_epochs=3,
            criterion=nn.CrossEntropyLoss(),
            train_split=ValidSplit(cv=0.2, stratified=True, random_state=5), # set later In case of intraDataset 
            verbose=0,
            callbacks=[EarlyStopping(patience=3)], 
            device=device
        ),
        # preprocessing_steps=[('preprocessing', TruncatedSVD()),('StandardScaler', StandardScaler())],
        preprocessing_steps=[('StandardScaler', StandardScaler(with_mean=False))],
        # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        tuning_space=tuning_space,
        # data_shape_required=True    
)


# Please don't change this line
wrapper = WrapperNN(**params)


if __name__ == "__main__":
    pass
