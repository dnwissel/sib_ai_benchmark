import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import optim

from .methods import TemperatureScaling, VectorScaling

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def train_model(model, input_logits, target, criterion, optimizer, epochs):
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(input_logits)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return model


def train_model_lbfgs(model, input_logits, target, criterion, lr):
    optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=1000)

    def closure():
        optimizer.zero_grad()
        output = model(input_logits)
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
        method='TS',
        encoder=None,
        lr=0.01
    ):
        self.classifier = model
        self.method_name = method
        self.criterion = criterion
        self.temperature = None
        self.encoder = encoder
        self.lr = lr
        self.method_trained = None

    def set_model(self, model):
        self.classifier = model

    def fit(self, X, y):
        logits = self.classifier.get_logits(X)
        if self.method_name == 'TS':
            method = TemperatureScaling().to(device)
        elif self.method_name == 'VS':
            method = VectorScaling(logits.shape[1]).to(device)
        else:
            raise ValueError('Invalid method name.')

        with torch.no_grad():
            if torch.is_tensor(logits):
                input_logits = logits.to(device)
            else:
                input_logits = torch.from_numpy(
                    logits).to(torch.float).to(device)

            if isinstance(y, pd.Series):
                target = torch.from_numpy(y.to_numpy()).to(device)
            else:
                target = torch.from_numpy(y).to(device)

        self.method_trained = train_model_lbfgs(
            method, input_logits, target, self.criterion, self.lr)
        return self

    # #TODO: argmax for PS/VS
    def get_logits(self, X):
        logits = self.classifier.get_logits(X)
        if torch.is_tensor(logits):
            input_logits = logits
        else:
            input_logits = torch.from_numpy(logits).float().to(device)

        output = self.method_trained(input_logits)

        if self.method_name == 'TS':
            params = list(self.method_trained.parameters())
            T = params[0]
            assert T > 0, f"Get negative temperature {T}"

        return output
