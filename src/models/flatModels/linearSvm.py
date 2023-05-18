from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

import numpy as np

from ..wrapper import Wrapper

params = dict(
        name='LinearSVM',
        model=LinearSVC(max_iter=10**6, tol=1e-3),
        preprocessing_steps=[('preprocessing', TruncatedSVD()),('StandardScaler', StandardScaler())],
        preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        tuning_space={
                'C': np.arange(0.5, 40, 0.5)  # C: Penalty parameter in Soft margin SVM
        }       
)


# Please don't change this line
wrapper = Wrapper(**params)


if __name__ == "__main__":
    pass
