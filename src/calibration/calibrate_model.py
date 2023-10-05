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
    def __init__(self, classifier, method='TS', type='flat', encoder=None):
        self.classifier = classifier
        self.method = method
        self.temperature = None
        self.model = None
        self.type = type
        self.encoder =encoder

    def fit(self, X, y): 
        if self.encoder is not None:
            y = self.encoder.transform(y)

        _, logits = self.classifier.predict_proba(self.classifier.model_fitted, X)
        if self.type == 'flat':
            criterion = nn.CrossEntropyLoss()
            model = TemperatureScaling().to(device)
        elif self.type == 'hier':
            criterion = F.binary_cross_entropy_with_logits
            model = VectorScaling(logits.shape[1]).to(device)
        else:
            raise ValueError('Invalid type.')

        optimizer = optim.Adam(model.parameters(), lr=0.05)
        
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
        # self.model = train_model(model, input, target, criterion, optimizer, 30)
        self.model = train_model_lbfgs(model, input, target, criterion)
        # print(self.model.temperature)
        return self

    #TODO: argmax for PS/VS
    def predict(self, X):
        preds = self.classifier.model_fitted.predict(X)
        if torch.is_tensor(preds):
            return preds.numpy().astype(int)
        return preds.astype(int)

    def predict_proba(self, X):
        _, logits = self.classifier.predict_proba(self.classifier.model_fitted, X)

        if torch.is_tensor(logits):
            input = logits
        else:
            input = torch.from_numpy(logits).to(torch.float).to(device)
            
        output = self.model(input)
        if self.type == 'flat':
            output = F.softmax(output, dim=-1)
        elif self.type == 'hier':
            # print('Enter')
            output = F.sigmoid(output)
            output = output.cpu().detach().numpy().astype(float)
            # output =  self.run_IR(output)
        return output

    #TODO: refactor
    def run_IR(self, probas):
        """ ref to https://qpsolvers.github.io/qpsolvers/quadratic-programming.html"""
        nodes = self.encoder.G_idx.nodes()
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
            x = solve_qp(0.5 * P, q, G=G, h=h,lb=lb, ub=ub, solver="osqp")
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
            successors = list(self.encoder.G_idx.successors(i))
            for child in successors:
                row = np.zeros(num_nodes)
                row[i] = 1.0
                row[child] = -1.0
                C.append(row)
        # print(np.array(C).shape, num_nodes)
        return np.array(C)