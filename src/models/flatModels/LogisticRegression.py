from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform
from scipy.special import softmax

import numpy as np

from ..wrapper import Wrapper

params = dict(
        name='LogisticRegression',
        model=LogisticRegression(random_state=0, max_iter=1000),
        preprocessing_steps=[('StandardScaler', StandardScaler())],
        # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        tuning_space={
                'class_weight':['balanced', None]
        }       
)


# Please don't change this line
wrapper = Wrapper(**params)


if __name__ == "__main__":
    pass
