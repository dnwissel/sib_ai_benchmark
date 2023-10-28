import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from .methods import TemperatureScaling, VectorScaling

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import softmax
import torch
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd

from qpsolvers import solve_qp
from scipy import sparse

from loss.hier import MCLoss, get_constr_out, MaskBCE

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# TODO : Dataloader
def train_model(model, input, target, criterion, optimizer, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        # print(f"Epoch {epoch}, Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
    return model
    

def train_model_lbfgs(model, input, target, criterion):
    optimizer = optim.LBFGS(model.parameters(), lr=0.01, max_iter=1000)

    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    return model


class CalibratedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self, 
            model=None, 
            criterion=None, 
            # optimizer=optim.Adam(model.parameters(), lr=0.5), 
            method='TS', 
            encoder=None,
            lr=0.01
        ):
        self.classifier = model
        self.method_name = method
        self.criterion=criterion
        self.temperature = None
        self.encoder = encoder
        self.lr = lr
        self.method_trained = None

    def set_model(self, model):
        self.classifier = model

    def fit(self, X, y): 
        # print(y)
        logits = self.classifier.get_logits(X)
        if self.method_name == 'TS':
            method = TemperatureScaling().to(device)
        elif self.method_name == 'VS':
            method = VectorScaling(logits.shape[1]).to(device)
        else:
            raise ValueError('Invalid method name.')

        self.optimizer = optim.Adam(method.parameters(), lr=self.lr)

        
        with torch.no_grad():
            if torch.is_tensor(logits):
                input = logits.to(device)
            else:
                input = torch.from_numpy(logits).to(torch.float).to(device)

            if isinstance(y, pd.Series):
                target = torch.from_numpy(y.to_numpy()).to(device)
            else:
                # y = y.astype(np.float32)
                target = torch.from_numpy(y).to(device)

        # target = F.one_hot(target, num_classes=logits.shape[1])
        # target = target.float()

        # print(input.device, target.device)
        # self.method = train_model(method, input, target, self.criterion, self.optimizer, 50)
        self.method_trained = train_model_lbfgs(method, input, target, self.criterion)
        return self

    # #TODO: argmax for PS/VS
    # def predict(self, X):
    #     preds = self.classifier.model_fitted.predict(X)
    #     if torch.is_tensor(preds):
    #         return preds.numpy().astype(int)
    #     return preds.astype(int)

    def get_logits(self, X):
        logits = self.classifier.get_logits(X)
        # print(logits)
        if torch.is_tensor(logits):
            input = logits
        else:
            input = torch.from_numpy(logits).float().to(device)
        
        output = self.method_trained(input)

        if self.method_name == 'TS':
            params = list(self.method_trained.parameters())
            T = params[0]
            assert T > 0, f"Get negative temperature {T}"
        
        return output

    

