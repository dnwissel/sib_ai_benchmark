import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from .temperature_scale import ModelWithTemperature

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import softmax


class CalibratedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier, method='TS'):
        self.classifier = classifier
        self.method = method
        self.temperature = None
        

    def fit(self, X, y):
        _, logits = self.classifier.predict_proba(self.classifier.model_fitted, X)
        ts = ModelWithTemperature()
        self.temperature = ts.set_temperature(logits, y)
        return self

    #TODO: argmax for PS
    def predict(self, X):
        return self.classifier.model_fitted.predict(X)

    def predict_proba(self, X):
        _, logits = self.classifier.predict_proba(self.classifier.model_fitted, X)
        return softmax(logits/self.temperature, axis=1)
