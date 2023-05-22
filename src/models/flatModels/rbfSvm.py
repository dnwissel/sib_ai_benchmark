from sklearn.svm import SVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform

import numpy as np
from ..wrapper import Wrapper

params = dict(
        name='RBFSVM', 
        model=SVC(kernel='rbf'),
        preprocessing_steps=[('preprocessing', TruncatedSVD()),('StandardScaler', StandardScaler())],
        preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        tuning_space = {
                'C': loguniform(1e-1, 1e3),  # C: Penalty parameter in Soft margin SVM
                'gamma': loguniform(1e-4, 1e-3),
                # 'gamma': ['scale', 'auto'] + np.arange(0.1, 30, 0.5).tolist(),
                'class_weight':['balanced', None]
        }
)

# Please don't change this line
wrapper = Wrapper(**params)


if __name__ == "__main__":
    pass
