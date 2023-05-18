from sklearn.svm import SVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

import numpy as np
from ..wrapper import Wrapper

params = dict(
        name='RBFSVM', 
        model=SVC(kernel='rbf'),
        preprocessing_steps=[('preprocessing', TruncatedSVD()),('StandardScaler', StandardScaler())],
        preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        tuning_space = {
                'C': np.arange(0.5, 40, 0.5),  # C: Penalty parameter in Soft margin SVM
                'gamma': ['scale', 'auto'] + np.arange(0.1, 30, 0.5).tolist()
        }
)

# Please don't change this line
wrapper = Wrapper(**params)


if __name__ == "__main__":
    pass
