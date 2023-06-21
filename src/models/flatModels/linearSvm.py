from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform

import numpy as np

from ..wrapper import Wrapper

params = dict(
        name='LinearSVM',
        model=LinearSVC(max_iter=10**6, tol=1e-3, class_weight='balanced', C=0.2037),
        preprocessing_steps=[('preprocessing', TruncatedSVD(n_components=60)),('StandardScaler', StandardScaler())],
        # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        # tuning_space={
        #         'C': loguniform(1e-1, 1e3),  # C: Penalty parameter in Soft margin SVM
        #         'class_weight':['balanced', None]
        # }       
)
# params = dict(
#         name='LinearSVM',
#         model=LinearSVC(max_iter=10**6, tol=1e-3),
#         preprocessing_steps=[('preprocessing', TruncatedSVD()),('StandardScaler', StandardScaler())],
#         preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
#         tuning_space={
#                 'C': loguniform(1e-1, 1e3),  # C: Penalty parameter in Soft margin SVM
#                 'class_weight':['balanced', None]
#         }       
# )


# Please don't change this line
wrapper = Wrapper(**params)


if __name__ == "__main__":
    pass
