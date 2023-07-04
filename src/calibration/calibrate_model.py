import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from .methods import TemperatureScaling, VectorScaling

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import softmax
import torch
from torch import nn, optim
from torch.nn import functional as F

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def train_model(model, input, target, criterion, optimizer, epochs):

    for epoch in range(epochs):
        output = model(input)
        loss = criterion(output, target)
        print(f"Epoch {epoch}, Loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
    

class CalibratedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier, method='TS'):
        self.classifier = classifier
        self.method = method
        self.temperature = None
        self.model = None
        

    def fit(self, X, y): 
        _, logits = self.classifier.predict_proba(self.classifier.model_fitted, X)
        # model = TemperatureScaling().to(device)
        model = VectorScaling(logits.shape[1]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.05)
        
        with torch.no_grad():
            input = torch.from_numpy(logits).to(torch.float).to(device)
            target = torch.from_numpy(y.to_numpy()).to(device)
        self.model = train_model(model, input, target, criterion, optimizer, 20)
        return self

    #TODO: argmax for PS/vs
    def predict(self, X):
        return self.classifier.model_fitted.predict(X)

    def predict_proba(self, X):
        _, logits = self.classifier.predict_proba(self.classifier.model_fitted, X)
        with torch.no_grad():
            input = torch.from_numpy(logits).to(torch.float).to(device)
        return self.model(input).detach().numpy().astype(np.float)
        # return softmax(logits/self.temperature, axis=1)
