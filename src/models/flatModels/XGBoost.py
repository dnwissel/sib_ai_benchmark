from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn

from models.wrapper import WrapperXGB
from calibration.calibrate_model import CalibratedClassifier


params = dict(
    name='XGBoost',
    model=HistGradientBoostingClassifier(
        validation_fraction=None, max_leaf_nodes=90, max_iter=100),
    preprocessing_steps=[('StandardScaler', StandardScaler())],
    tuning_space={
        'learning_rate': loguniform(1e-3, 5e-1),
        'class_weight': ['balanced', None]
    },
    calibrater=CalibratedClassifier(
        criterion=nn.CrossEntropyLoss(), method='VS', lr=0.01)
)


# Please don't change this line
wrapper = WrapperXGB(**params)


if __name__ == "__main__":
    pass
