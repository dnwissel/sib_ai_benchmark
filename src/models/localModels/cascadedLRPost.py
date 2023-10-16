import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
import networkx as nx
from models.wrapper import WrapperLocal
from sklearn.base import clone
from inference.infer import infer_1, infer_2
from sklearn.preprocessing import StandardScaler



class CascadedLRPost:
    def __init__(
            self,
            encoder=None,
            base_learner=LogisticRegression(max_iter=1000, class_weight='balanced'),
        ):
        self.encoder = encoder
        self.trained_classifiers = None
        self.base_learner = base_learner
        self.node_indicators = None
        self.path_eval = False

    def set_predictPath(self, val):
        self.path_eval = val
       
    def fit(self, X, y):
        self._fit_base_learner(X, y)
        return self
    
    def _fit_base_learner(self, X, y):
        encoded_y = self.encoder.transform(y)
        self.node_indicators = encoded_y.T
        self.trained_classifiers = np.zeros(encoded_y.shape[1], dtype=object)
        for idx, node_y in enumerate(self.node_indicators):
            unique_node_y = np.unique(node_y)
            if len(unique_node_y) == 1:
                cls = unique_node_y[0]
            else:
                cls = clone(self.base_learner)
                cls = cls.fit(X, node_y)
            self.trained_classifiers[idx] = cls


    def _compute_marginals(self, anc_matrix, log_marginal_probas, log_probas, idx):
        log_proba_current = log_probas[idx]
        log_mp = log_marginal_probas[idx] 
        if log_mp is not None:
            return log_mp
        
        anc_mask = anc_matrix[idx]
        sum = 0
        for idx_anc, val in enumerate(anc_mask):
            if val == 0:
                continue
            anc_proba = log_marginal_probas[idx_anc]
            if anc_proba is not None:
                sum += anc_proba
            else:
                # log_proba_anc = log_probas[idx_anc]
                sum += self._compute_marginals(anc_matrix, log_marginal_probas, log_probas, idx_anc)
        log_marginal_probas[idx] = sum + log_proba_current
        # print(type(log_marginal_probas[idx]))
        return sum


    def get_marginal_proba(self, log_probas_full):
        """Algo in CELLO paper, actually AND logic"""
        # Compute the marginals for each label by multiplying all of the conditional
        # probabilities from that label up to the root of the DAG
        anc_matrix = self.encoder.get_ancestorMatrix()
        anc_matrix = anc_matrix.astype(int)

        marginal_probas_full = []
        for log_probas in log_probas_full:
            # print(log_probas)
            log_marginal_probas = np.full(shape=len(log_probas), fill_value=None)
            roots_idx = self.encoder.roots_idx
            log_marginal_probas[roots_idx] = 0.0
            # log_marginal_probas = log_marginal_probas.tolist()
            for idx in range(len(log_probas)):
                self._compute_marginals(anc_matrix, log_marginal_probas, log_probas, idx)
                # print(log_marginal_probas[idx])
            
            # print(log_marginal_probas)
            log_marginal_probas = log_marginal_probas.astype(float)
            marginal_probas = np.exp(log_marginal_probas)
            marginal_probas_full.append(marginal_probas)
        return np.array(marginal_probas_full)

    def set_encoder(self, encoder):
        self.encoder = encoder


    def predict_log_proba(self, X):
        probas = []
        # To avoid log(0)
        epsilon = 1e-10
        for cls in self.trained_classifiers:
            if isinstance(cls, float):
                if cls == 0.0:
                    cls += epsilon
                col = np.repeat(np.log([cls]), X.shape[0]) # TODO log
            else:
                probas_all = cls.predict_proba(X) # proba for pos class
                col = probas_all[:, 1] # proba for pos class

                mask = (col == 0.0)
                col[mask] += epsilon
                col = np.log(col)
            probas.append(col)
        probas =  np.array(probas).T
        # print(None in probas.flatten())
        return probas

    def predict(self, X, threshold=0.5):
        log_probas = self.predict_log_proba(X)
        marginal_probas_full = self.get_marginal_proba(log_probas)
        if self.path_eval:
            return marginal_probas_full > threshold
        return infer_1(marginal_probas_full, self.encoder)

    def predict_proba(self, X):
        log_probas = self.predict_log_proba(X)
        marginal_probas_full = self.get_marginal_proba(log_probas)
        return marginal_probas_full, None

        
params = dict(
        name='CascadedLRPost',
        model=CascadedLRPost(),
        preprocessing_steps=[('StandardScaler', StandardScaler())]
)

# Please don't change this line
wrapper = WrapperLocal(**params)


if __name__ == "__main__":
    pass