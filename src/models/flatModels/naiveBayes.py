from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB


from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform

import numpy as np

from ..wrapper import Wrapper

params = dict(
        name='NaiveBayes',
        # model=ComplementNB(force_alpha=True),
        model=MultinomialNB(force_alpha=True),
        # model=GaussianNB(),
        # preprocessing_steps=[('preprocessing', TruncatedSVD(n_components=60))],
        # preprocessing_steps=[('preprocessing', TruncatedSVD(n_components=60)),('StandardScaler', StandardScaler())],
        # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        # tuning_space={
        #         'C': loguniform(1e-1, 1e3),  # C: Penalty parameter in Soft margin SVM
        #         'class_weight':['balanced', None]
        # }       
)

# Please don't change this line
wrapper = Wrapper(**params)


if __name__ == "__main__":
    pass