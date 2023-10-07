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
    optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=1000)

    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    return model


class CalibratedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier, criterion=None, optimizer=None, method='TS', type='hier', encoder=None):
        self.classifier = classifier
        self.method = method
        self.temperature = None
        self.model = None
        self.type = type
        self.encoder = encoder

    def fit(self, X, y): 
        # if self.encoder is not None:
            # y = self.encoder.transform(y)

        _, logits = self.classifier.predict_proba(X)
        if self.type == 'flat':
            criterion = nn.CrossEntropyLoss()
            model = TemperatureScaling().to(device)
        elif self.type == 'hier':
            # base_criterion = F.binary_cross_entropy_with_logits
            criterion =  MCLoss(self.encoder)
            criterion =  MaskBCE(self.encoder)
            # y = self.encoder.transform(y)
            # criterion =  nn.BCELoss()

            model = VectorScaling(logits.shape[1]).to(device)
        else:
            raise ValueError('Invalid type.')

        optimizer = optim.Adam(model.parameters(), lr=0.5)
        
        with torch.no_grad():
            if torch.is_tensor(logits):
                input = logits.to(device)
            else:
                input = torch.from_numpy(logits).to(torch.float).to(device)

            if isinstance(y, pd.Series):
                target = torch.from_numpy(y.to_numpy()).to(device)
            else:
                target = torch.from_numpy(y).to(device)

        # print(input.device, target.device)
        self.model = train_model(model, input, target, criterion, optimizer, 30)
        # self.model = train_model_lbfgs(model, input, target, criterion)
        # print(self.model.temperature)
        return self

    #TODO: argmax for PS/VS
    def predict(self, X):
        preds = self.classifier.model_fitted.predict(X)
        if torch.is_tensor(preds):
            return preds.numpy().astype(int)
        return preds.astype(int)

    def predict_proba(self, X):
        _, logits = self.classifier.predict_proba(X)

        if torch.is_tensor(logits):
            input = logits
        else:
            input = torch.from_numpy(logits).to(torch.float).to(device)
            
        output = self.model(input)

        if self.type == 'flat':
            output = F.softmax(output, dim=-1)
        # elif self.type == 'hier':
        #     # print('Enter')
        #     output = get_constr_out(output, self.encoder.get_R())
        #     output = output.cpu().detach().numpy().astype(float)
        return output

    

