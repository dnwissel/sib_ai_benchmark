import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from sklearn.isotonic import isotonic_regression
# from quadprog import solve_qp
from qpsolvers import solve_qp
from scipy import sparse

import networkx as nx
from models.wrapper import WrapperLocal
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from models.baseModel import LocalModel
from inference.infer import infer_1, infer_2




class IsotonicRegressionPost(LocalModel):
        
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


    def predict_proba(self, X, raw=False):
        probas = []
        for cls in self.trained_classifiers:
            if isinstance(cls, float):
                col = np.repeat([cls], X.shape[0])
            else:
                col = cls.predict_proba(X)[:, 1] # proba for pos class
            probas.append(col)
        if raw:
            return np.array(probas).T

        probas =  self.run_IR(np.array(probas).T)
        return probas, None


    def predict(self, X, threshold=0.5):
        probas, _ = self.predict_proba(X)
        if self.path_eval:
            return probas > threshold
            
        return infer_1(probas, self.encoder)


params = dict(
        name='IsotonicRegressionPost',
        model=IsotonicRegressionPost(
            LogisticRegression(max_iter=1000, class_weight='balanced')
        ),
        preprocessing_steps=[('StandardScaler', StandardScaler())]
)

# Please don't change this line
wrapper = WrapperLocal(**params)


if __name__ == "__main__":
    pass