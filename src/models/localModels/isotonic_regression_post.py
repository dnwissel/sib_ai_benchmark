#################################################################
#   Hierarchical classification via isotonic regression
#################################################################
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
# from sklearn.isotonic import isotonic_regression
from quadprog import solve_qp

from utilities.hier import get_R

class IsotonicRegressionPost():
    def __init__(
            self,
            base_learner=LogisticRegression()
            encoder
        ):
        self.encoder = encoder
        self.trained_classifiers = None
        self.base_learner = base_learner
       
    def fit(self, X, y):
        pass
    
    def fit_base_learner(self, X, y):
        encoded_y = self.encoder.transform(y)
        R = get_R(self.encoder)
        node_indicators = encoded_y.T
        self.trained_classifiers = np.zeros(encoded_y.shape[1])
        for idx, node_y in enumerator(node_indicators):
            unique_node_y = np.unique(node_y)
            if len(unique_node_y) == 1:
                self.trained_classifiers[idx] = unique_node_y[0]
            else:
                self.trained_classifiers[idx] = self.base_learner.fit(X, node_y)

    def run_IR(self):
        pass
       
    def get_C(self):
        pass

    def predict_proba(self):
        pass

    def predict(self, X):
        pass
        