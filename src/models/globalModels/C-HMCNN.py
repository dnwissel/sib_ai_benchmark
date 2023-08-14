import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np

from models.wrapper import WrapperHier
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from scipy.stats import loguniform, uniform, randint


def get_constr_out(x, R):
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x),R.shape[1], R.shape[1])
    R_batch = R.expand(len(x),R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch*c_out.double(), dim = 2)
    return final_out

class MCLoss(nn.Module):
    def __init__(self, R, idx_to_eval):
        super().__init__()
        self.R = R
        self.idx_to_eval = idx_to_eval
        self.criterion = nn.BCELoss()


    def forward(self, output, target):
        constr_output = get_constr_out(output, self.R)
        train_output = target*output.double()
        train_output = get_constr_out(train_output, self.R)
        train_output = (1-target)*constr_output.double() + target*train_output

        #MCLoss
        loss = self.criterion(train_output[:,self.idx_to_eval ], target[:,self.idx_to_eval])
        return loss


class C_HMCNN(nn.Module):
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

    #TODO: Refine 
    def forward(self, X, **kwargs):
        x = self.layers_seq(X)
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out


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

model=NeuralNetClassifier(
            module=C_HMCNN,
            # max_epochs=30,
            max_epochs=3,
            criterion=MCLoss,
            train_split=ValidSplit(cv=0.2, stratified=True, random_state=5), # set later In case of intraDataset 
            verbose=0,
            callbacks=[EarlyStopping(patience=3)], 
            device=device
        )

params = dict(
        name='C-HMCNN',
        model=model,
        # preprocessing_steps=[('preprocessing', TruncatedSVD()),('StandardScaler', StandardScaler())],
        preprocessing_steps=[('StandardScaler', StandardScaler(with_mean=False))],
        # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        tuning_space=tuning_space,
        # data_shape_required=True    
)


# Please don't change this line
wrapper = WrapperHier(**params)


if __name__ == "__main__":
    pass
