from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform

import numpy as np

from models.wrapper import WrapperSVM
from calibration.calibrate_model import CalibratedClassifier
from torch import nn



# params = dict(
#         name='LinearSVM',
#         model=LinearSVC(max_iter=10**6, tol=1e-3, class_weight='balanced', C=0.2037),
#         preprocessing_steps=[('StandardScaler', StandardScaler())],
#         # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
#         # tuning_space={
#         #         'C': loguniform(1e-1, 1e3),  # C: Penalty parameter in Soft margin SVM
#         #         'class_weight':['balanced', None]
#         # }       
# )
params = dict(
        name='LinearSVM',
        model=LinearSVC(max_iter=10**4, tol=1e-3, dual='auto'),
        preprocessing_steps=[('StandardScaler', StandardScaler())],
        # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        tuning_space={
                'C': loguniform(1e-1, 1e1),  # C: Penalty parameter in Soft margin SVM
                'class_weight':['balanced', None]
        },
        calibrater=CalibratedClassifier(criterion=nn.CrossEntropyLoss(), method='VS', lr=0.01)
    
)


# Please don't change this line
wrapper = WrapperSVM(**params)


if __name__ == "__main__":
    pass
