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

from models.wrapper import WrapperCHMC, WrapperCS
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from scipy.stats import loguniform, uniform, randint

from scipy import sparse
from qpsolvers import solve_qp
import matplotlib.pyplot as plt

from loss.hier import MaskBCE
from calibration.calibrate_model import CalibratedClassifier
from inference import infer
from utilities.customizedValidSplit import CustomizedValidSplit
from models.baseModel import MLP, tuning_space



class NeuralNetClassifierHier_2(NeuralNetClassifier):
    def set_predictPath(self, val):
        self.path_eval = val
    
    def predict(self, X, threshold=0.5):
        if hasattr(self, 'path_eval') and self.path_eval:
            probas, _ = self.predict_proba(X)
            return probas > threshold

        output = self.forward(X)
        probas = torch.sigmoid(output) 
        probas = probas.to('cpu').numpy()
        preds, _ = infer.infer_cs(probas, self.module.en)
        return preds

    def predict_proba(self, X):
        output = self.forward(X)
        probas = torch.sigmoid(output) 
        probas = probas.to('cpu').numpy()
        
        if hasattr(self, 'path_eval') and self.path_eval:
            probas = infer.infer_path_cs(probas, self.module.en)
            probas_consitant = infer.run_IR(probas, self.module.en)
            return probas_consitant, output

        _, probas_consitant = infer.infer_cs(probas, self.module.en)
        return probas_consitant, output 


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model=NeuralNetClassifierHier_2(
            module=MLP,
            # max_epochs=30,
            max_epochs=1 if cfg.debug else 20,
            criterion=MaskBCE,
            # train_split=CustomizedValidSplit(cv=0.1, stratified=True, random_state=5), 
            train_split=None,
            # train_split=ValidSplit(cv=0.2, stratified=True, random_state=5), 
            verbose=0,
            callbacks=[EarlyStopping(patience=5, monitor='train_loss')], 
            # warm_start=False,
            device=device
        )

params = dict(
        name='ConditionalSigmoid',
        model=model,
        # preprocessing_steps=[('preprocessing', TruncatedSVD()),('StandardScaler', StandardScaler())],
        # preprocessing_steps=[('StandardScaler', StandardScaler(with_mean=False))],
        preprocessing_steps=[('StandardScaler', StandardScaler())],
        # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        tuning_space=tuning_space,
        calibrater=CalibratedClassifier(criterion=MaskBCE(), method='VS')
        # calibrater=CalibratedClassifier(criterion=F.binary_cross_entropy_with_logits, method='VS')

)


# Please don't change this line
wrapper = WrapperCS(**params)


if __name__ == "__main__":
    pass
