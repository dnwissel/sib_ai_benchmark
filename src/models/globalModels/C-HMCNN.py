import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np
import networkx as nx
from config import cfg

from models.wrapper import WrapperCHMC
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from scipy.stats import loguniform, uniform, randint

from calibration.calibrate_model import CalibratedClassifier
from loss.hier import MCLoss, get_constr_out
from inference import infer


class NeuralNetClassifierHier_1(NeuralNetClassifier):
    def set_predictPath(self, val):
        self.path_eval = val
        
    def predict(self, X, threshold=0.5):
        output = self.forward(X)
        constrained_out = get_constr_out(output, self.module.en.get_R())
        constrained_out = constrained_out.to('cpu')

        if hasattr(self, 'path_eval') and self.path_eval:
            return constrained_out > threshold
            
        preds = infer.infer_1(constrained_out, self.module.en)
        return preds

    def predict_proba(self, X):
        output = self.forward(X)
        constrained_out = get_constr_out(output, self.module.en.get_R())
        constrained_out = constrained_out.to('cpu')
        probas = F.sigmoid(constrained_out).numpy()

        probas[:, self.module.en.roots_idx] = 1.0
        # logits = constrained_out.numpy()
        logits = constrained_out
        return probas, logits


class C_HMCNN(nn.Module):
    def __init__(self, dim_in, dim_out, nonlin, num_hidden_layers,  dor_input, dor_hidden, neuron_power, en):
        super().__init__()
        # self.module.en = en
        C_HMCNN.en = en
        # self.R = R

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
        # if self.training:
        #     return x
        # constrained_out = get_constr_out(x, self.R)
        # constrained_out = self._inference(constrained_out)
        # return constrained_out
        return x


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


tuning_space={
                'lr': loguniform(1e-4, 1e-3),
                'batch_size': [4, 16, 32],
                # 'optimizer': [optim.SGD, optim.Adam],
                'optimizer': [optim.Adam],
                'optimizer__weight_decay': loguniform(1e-5, 1e-4),
                # 'optimizer__momentum': loguniform(1e-3, 1e0),
                # 'module__nonlin': [nn.ReLU, nn.Tanh, nn.Sigmoid],
                'module__nonlin': [nn.ReLU],
                # 'module__num_hidden_layers': np.arange(0 , 8 , 2).tolist(),
                'module__num_hidden_layers': [1],
                # 'module__dor_input': uniform(0, 0.3),
                'module__neuron_power': range(9, 11),
                'module__dor_input': [0],
                'module__dor_hidden': uniform(0, 1)
}

model=NeuralNetClassifierHier_1(
            module=C_HMCNN,
            # max_epochs=30,
            max_epochs=1 if cfg.debug else 30,
            criterion=MCLoss,
            train_split=None,
            # train_split=ValidSplit(cv=0.2, stratified=True, random_state=5), # set later In case of intraDataset 
            verbose=0,
            # callbacks=[EarlyStopping(patience=3)], 
            # warm_start=False,
            device=device
        )


params = dict(
        name='C-HMCNN',
        model=model,
        # preprocessing_steps=[('preprocessing', TruncatedSVD()),('StandardScaler', StandardScaler())],
        preprocessing_steps=[('StandardScaler', StandardScaler())],
        # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        tuning_space=tuning_space,
        # data_shape_required=True 
        calibrater=CalibratedClassifier(criterion=MCLoss(), method='VS')
)


# Please don't change this line
wrapper = WrapperCHMC(**params)


if __name__ == "__main__":
    pass