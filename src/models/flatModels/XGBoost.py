from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform
from sklearn.ensemble import HistGradientBoostingClassifier

import numpy as np

from models.wrapper import Wrapper


params = dict(
        name='XGBoost',
        model=HistGradientBoostingClassifier(validation_fraction=None, max_iter=30),
        preprocessing_steps=[('StandardScaler', StandardScaler())],
        # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        tuning_space={
                'learning_rate': loguniform(1e-3, 1e0),  
                'class_weight':['balanced', None]
        }       
)


# Please don't change this line
wrapper = Wrapper(**params)


if __name__ == "__main__":
    pass
