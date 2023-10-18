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
from utilities.customizedValidSplit import CustomizedValidSplit
from models.baseModel.net import MLP, tuning_space


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



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


model=NeuralNetClassifierHier_1(
            module=MLP,
            # max_epochs=30,
            max_epochs=1 if cfg.debug else 20,
            criterion=MCLoss,
            train_split=None,
            # train_split=CustomizedValidSplit(cv=0.1, stratified=True, random_state=5), 
            # train_split=ValidSplit(cv=0.2, stratified=True, random_state=5), 
            verbose=0,
            # callbacks=[EarlyStopping(patience=5, monitor='train_loss')], 
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
        
        calibrater=CalibratedClassifier(criterion=MCLoss(), method='VS')
)


# Please don't change this line
wrapper = WrapperCHMC(**params)


if __name__ == "__main__":
    pass