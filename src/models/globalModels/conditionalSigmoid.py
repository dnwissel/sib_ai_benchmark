from functools import partial
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

from models.wrapper import WrapperHier, WrapperHierCS
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from scipy.stats import loguniform, uniform, randint

from scipy import sparse
from qpsolvers import solve_qp


class NeuralNetClassifierHier_2(NeuralNetClassifier):

    def set_predictPath(self, val):
        self.predict_path = val
    
    def predict(self, X, threshold=0.5):
        output = self.forward(X)
        probas = torch.sigmoid(output) 
        probas = probas.to('cpu').numpy()
        
        if hasattr(self, 'predict_path') and self.predict_path:
            probas_consitant = self.run_IR(probas)
            return self._inference_path(probas) > threshold

        preds = self._inference(probas)
        return preds
    
    def run_IR(self, probas):
            """ 
            ref to https://qpsolvers.github.io/qpsolvers/quadratic-programming.html
            """

            nodes = self.module.en.G_idx.nodes()
            num_nodes = len(nodes)
            P = np.zeros((num_nodes, num_nodes))
            np.fill_diagonal(P, 1)

            C = self._get_C(nodes)
            G = sparse.csc_matrix(-1 * C)
            P = sparse.csc_matrix(P)

            h = np.zeros(C.shape[0])
            lb = np.zeros(C.shape[1])
            ub = np.ones(C.shape[1])
            probas_post = []
            for row in probas:
                q = -1 * row.T
                x = solve_qp(P, q, G=G, h=h,lb=lb, ub=ub, solver="osqp")
                probas_post.append(x)
            # print(x - row)
            return np.array(probas_post)


    def _get_C(self, nodes):
        """
        Constraint matrix for quadratic prog, ensure that, pi < pj, if j is a parent of i.
        """
        num_nodes = len(nodes)
        C = []
        for i in range(num_nodes):
            successors = list(self.module.en.G_idx.successors(i))
            for child in successors:
                row = np.zeros(num_nodes)
                row[i] = 1.0
                row[child] = -1.0
                C.append(row)
        # print(np.array(C).shape, num_nodes)
        return np.array(C)


    def _lhs_dp(self, node, en, row, memo):
        value = memo[node]
        if value != -1:
            return value
        s_prime_pos = list(map(partial(self._lhs_dp, en=en, row=row, memo=memo), en.predecessor_dict[node])) 
        lh = row[node] * (1 - torch.prod(1 - torch.tensor(s_prime_pos)))
        memo[node] = lh
        return memo[node]
        

    def _inference(self, probas):
        y_pred = np.zeros(probas.shape[0])
        for row_idx, row in enumerate(probas):
            memo = np.zeros(len(self.module.en.G_idx.nodes())) - 1
            for root in self.module.en.roots_idx:
                memo[root] = 1

            lhs = np.zeros(len(self.module.en.label_idx))
            for idx, label in enumerate(self.module.en.label_idx):
                lh_ = self._lhs_dp(label, self.module.en, row, memo)
                lh_children = np.prod(1 -  row[list(self.module.en.successor_dict[label])])
                lhs[idx] = lh_ * lh_children
            y_pred[row_idx] = self.module.en.label_idx[np.argmax(lhs)]
        return y_pred

    def _inference_path(self, probas):
        y_pred = np.zeros(probas.shape)
        num_nodes = probas.shape[1]
        for row_idx, row in enumerate(probas):
            memo = np.zeros(num_nodes) - 1

            for root in self.module.en.roots_idx:
                memo[root] = 1

            lhs = np.zeros(num_nodes)
            for label in range(num_nodes):
                lh_ = self._lhs_dp(label, self.module.en, row, memo)
                lh_children = np.prod(1 -  row[list(self.module.en.successor_dict[label])])
                lhs[label] = lh_ * lh_children
            y_pred[row_idx] = lhs
        return y_pred


class MaskBCE(nn.Module):
    def __init__(self, en, idx_to_eval):
        super().__init__()
        self.en = en
        # self.R = R
        self.idx_to_eval = idx_to_eval
        # self.criterion = F.binary_cross_entropy()
        # self.bid = 0


    def forward(self, output, target):
        # constr_output = get_constr_out(output, self.R)
        # train_output = target*output.double()
        # train_output = get_constr_out(train_output, self.R)
        # train_output = (1-target)*constr_output.double() + target*train_output
        train_output = output

        #Mask Loss
        loss_mask = self.en.get_lossMask()
        loss_mask = loss_mask.to(device)
        lm_batch = loss_mask[target]
        target = self.en.transform(target.cpu().numpy())
        target = target.astype(np.float32)
        target = torch.from_numpy(target).to(device)

        # #Mask target
        # lm_batch = loss_mask[target]
        # target = self.en.transform(target.numpy())
        # target = target.astype(np.float32)
        # target = np.where(lm_batch, target, 1)
        # target = torch.from_numpy(target).to(device)

        loss = F.binary_cross_entropy_with_logits(train_output[:,self.idx_to_eval], target[:,self.idx_to_eval], reduction='none')
        loss = lm_batch[:,self.idx_to_eval] * loss
        return loss.sum()
    

class ConditionalSigmoid(nn.Module):
    def __init__(self, dim_in, dim_out, nonlin, num_hidden_layers,  dor_input, dor_hidden, neuron_power, en):
        super().__init__()
        # self.module.en = en
        ConditionalSigmoid.en = en
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

model=NeuralNetClassifierHier_2(
            module=ConditionalSigmoid,
            # max_epochs=30,
            max_epochs=1 if cfg.debug else 30,
            criterion=MaskBCE,
            train_split=None,
            # train_split=ValidSplit(cv=0.2, stratified=True, random_state=5), # set later In case of intraDataset 
            verbose=0,
            # callbacks=[EarlyStopping(patience=3)], 
            # warm_start=False,
            device=device
        )

params = dict(
        name='ConditionalSigmoid',
        model=model,
        # preprocessing_steps=[('preprocessing', TruncatedSVD()),('StandardScaler', StandardScaler())],
        preprocessing_steps=[('StandardScaler', StandardScaler(with_mean=False))],
        # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        tuning_space=tuning_space,
        # data_shape_required=True    
)


# Please don't change this line
wrapper = WrapperHierCS(**params)


if __name__ == "__main__":
    pass
