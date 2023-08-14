from sklearn.svm import SVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform
from scipy.special import softmax


import numpy as np
from models.wrapper import WrapperSVM


params = dict(
        name='RBFSVM', 
        model=SVC(kernel='rbf'),
        preprocessing_steps=[('StandardScaler', StandardScaler())],
        tuning_space = {
                'C': loguniform(1e-1, 1e3),  # C: Penalty parameter in Soft margin SVM
                'gamma': loguniform(1e-4, 1e-3),
                # 'gamma': ['scale', 'auto'] + np.arange(0.1, 30, 0.5).tolist(),
                'class_weight':['balanced', None]
        }
)

# Please don't change this line
wrapper = WrapperSVM(**params)


if __name__ == "__main__":
    pass
