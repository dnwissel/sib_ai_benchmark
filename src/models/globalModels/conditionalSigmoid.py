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

class NeuralNetClassifierHier(NeuralNetClassifier):

    def __init__(
            self,
            module,
            *args,
            criterion=torch.nn.NLLLoss,
            train_split=ValidSplit(5, stratified=True),
            classes=None,
            **kwargs
    ):
        super(NeuralNetClassifierHier, self).__init__(
            module,
            *args,
            criterion=criterion,
            train_split=train_split,
            **kwargs
        )
        # self.classes = classes
    
    def fit_loop(self, X, y=None, epochs=None, **fit_params): 
        # y = self.module.en.transform(y)  

        self.check_data(X, y)
        self.check_training_readiness()
        epochs = epochs if epochs is not None else self.max_epochs

        dataset_train, dataset_valid = self.get_split_datasets(
            X, y, **fit_params)
        on_epoch_kwargs = {
            'dataset_train': dataset_train,
            'dataset_valid': dataset_valid,
        }
        iterator_train = self.get_iterator(dataset_train, training=True)
        iterator_valid = None
        if dataset_valid is not None:
            iterator_valid = self.get_iterator(dataset_valid, training=False)

        for _ in range(epochs):
            self.notify('on_epoch_begin', **on_epoch_kwargs)

            self.run_single_epoch(iterator_train, training=True, prefix="train",
                                  step_fn=self.train_step, **fit_params)

            self.run_single_epoch(iterator_valid, training=False, prefix="valid",
                                  step_fn=self.validation_step, **fit_params)

            self.notify("on_epoch_end", **on_epoch_kwargs)
        return self


    def predict(self, X):
        output = self.forward(X)
        constrained_out = get_constr_out(output, self.module.R)
        preds = self._inference(constrained_out)
        return preds
    

    def _lhs(self, node, en, row, memo):
        value = memo[node]
        if value != -1:
            return value
        s_prime_pos = list(map(partial(self._lhs, en=en, row=row, memo=memo), en.predecessor_dict[node])) 
        lh_children = torch.prod(1 -  row[list(en.successor_dict[node])])
        lh = row[node] * (1 - torch.prod(1 - torch.tensor(s_prime_pos))) * lh_children
        memo[node] = lh
        return memo[node]


    def _inference(self, constrained_output):
        constrained_output = constrained_output.to('cpu')
        y_pred = np.zeros(constrained_output.shape[0])
        for row_idx, row in enumerate(constrained_output.data):
            # init graph node attribute
            # nx.set_node_attributes(en.G_idx, -1, name="lh")
            # for root in en.roots_idx:
            #     nx.set_node_attributes(en.G_idx, {root: {'lh': 1}})

            memo = np.zeros(self.module.R.shape[1]) - 1
            for root in self.module.en.roots_idx:
                memo[root] = 1

            lhs = np.zeros(len(self.module.en.label_idx))
            for idx, label in enumerate(self.module.en.label_idx):
                lh_ = self._lhs(label, self.module.en, row, memo)
                lhs[idx] = lh_
            y_pred[row_idx] = self.module.en.label_idx[np.argmax(lhs)]
        # y_true = [self.module.en.node_map.get(e) for e in test_Y_raw]
        # y_true = list(map(self.module.en.node_map.get, test_Y_raw))
        return y_pred


def get_constr_out(x, R):
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    # print(type(x))
    if type(x) is np.ndarray:
        print(x)
    
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x),R.shape[1], R.shape[1])
    R_batch = R.expand(len(x),R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch*c_out.double(), dim = 2)
    return final_out


class MaskBCE(nn.Module):
    def __init__(self, en, loss_mask, idx_to_eval):
        super().__init__()
        self.en = en
        self.loss_mask = loss_mask
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
        lm_batch = self.loss_mask[target]
        target = self.en.transform(target)
        target = target.astype(np.float32)
        target = torch.from_numpy(target).to(device)

        # lm_batch = self.loss_mask[self.label_loader[self.bid], :][:,  self.idx_to_eval]
        # print(lm_batch.shape, train_output.shape,target.shape)
        loss = F.binary_cross_entropy_with_logits(train_output[:,self.idx_to_eval], target[:,self.idx_to_eval], reduction='none')
        loss = lm_batch[:,self.idx_to_eval] * loss
        # self.bid = (self.bid + 1) % len(self.label_loader)
        # print(self.bid)
        return loss.sum()
    

class ConditionalSigmoid(nn.Module):
    def __init__(self, dim_in, dim_out, nonlin, num_hidden_layers,  dor_input, dor_hidden, neuron_power, en,  R):
        super().__init__()
        # self.module.en = en
        ConditionalSigmoid.en = en
        # self.R = R
        ConditionalSigmoid.R = R

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

model=NeuralNetClassifierHier(
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
