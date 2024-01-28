from scipy.stats import loguniform
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

from calibration.calibrate_model import CalibratedClassifier
from models.wrapper import WrapperSVM

params = dict(
    name='RBFSVM',
    model=SVC(kernel='rbf'),
    preprocessing_steps=[('StandardScaler', StandardScaler())],
    tuning_space={
        'C': loguniform(1e-1, 1e1),  # C: Penalty parameter in Soft margin SVM
        # 'gamma': loguniform(1e-4, 1e-3),
        'gamma': ['scale', 'auto'],
        'class_weight': ['balanced', None]
    },
    calibrater=CalibratedClassifier(
        criterion=nn.CrossEntropyLoss(), method='VS', lr=0.01)
    # calibrater=CalibratedClassifier(criterion=F.binary_cross_entropy_with_logits, method='VS')

)

# Please don't change this line
wrapper = WrapperSVM(**params)


if __name__ == "__main__":
    pass
