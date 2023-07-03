from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform
from scipy.special import softmax

import numpy as np

from ..wrapper import Wrapper

class WrapperSVM(Wrapper):
        def __init__(self, model, name, tuning_space=None, preprocessing_steps=None, preprocessing_params=None, is_selected=True, data_shape_required=False): 
                super().__init__(model, name, tuning_space, preprocessing_steps, preprocessing_params, is_selected, data_shape_required)

        def predict_proba(self, model_fitted, X):
                confidence = model_fitted.decision_function(X)
                if confidence.ndim == 1 or confidence.shape[1] == 1:
                        confidence = np.reshape(confidence, (-1, 1))
                        confidence = np.concatenate((-1 * confidence, 1 * confidence), axis=1) # label 1 considered as positive, 
                        # print(confidence)
                        return softmax(confidence, axis=1), confidence
                return softmax(confidence, axis=1),confidence

# params = dict(
#         name='LinearSVM',
#         model=LinearSVC(max_iter=10**6, tol=1e-3, class_weight='balanced', C=0.2037),
#         preprocessing_steps=[('preprocessing', TruncatedSVD(n_components=60)),('StandardScaler', StandardScaler())],
#         # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
#         # tuning_space={
#         #         'C': loguniform(1e-1, 1e3),  # C: Penalty parameter in Soft margin SVM
#         #         'class_weight':['balanced', None]
#         # }       
# )
params = dict(
        name='LinearSVM',
        model=LinearSVC(max_iter=10**6, tol=1e-3),
        # preprocessing_steps=[('preprocessing', TruncatedSVD()),('StandardScaler', StandardScaler())],
        # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
        tuning_space={
                'C': loguniform(1e-1, 1e3),  # C: Penalty parameter in Soft margin SVM
                'class_weight':['balanced', None]
        }       
)


# Please don't change this line
wrapper = WrapperSVM(**params)


if __name__ == "__main__":
    pass
