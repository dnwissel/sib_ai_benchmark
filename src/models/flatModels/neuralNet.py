import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from calibration.calibrate_model import CalibratedClassifier
from config import cfg
from models.baseModel import MLP, tuning_space
from models.wrapper import WrapperNN
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

params = dict(
    name='NeuralNet',
    model=NeuralNetClassifier(
        module=MLP,
        max_epochs=5 if cfg.debug else 20,
        criterion=nn.CrossEntropyLoss(),
        train_split=None,
        # train_split=CustomizedValidSplit(cv=0.1, stratified=True, random_state=5),
        # train_split=ValidSplit(cv=0.15, stratified=True, random_state=None),
        callbacks=[EarlyStopping(patience=5, monitor='train_loss')],
        # train_split=None,
        verbose=0,
        device=device
    ),
    # preprocessing_steps=[('preprocessing', TruncatedSVD()),('StandardScaler', StandardScaler())],
    # preprocessing_steps=[('StandardScaler', StandardScaler(with_mean=False))],
    preprocessing_steps=[('StandardScaler', StandardScaler())],
    # preprocessing_params = {'preprocessing__n_components': np.arange(10, 100, 10)},
    tuning_space=tuning_space,
    calibrater=CalibratedClassifier(
        criterion=nn.CrossEntropyLoss(), method='TS', lr=0.01)
)


# Please don't change this line
wrapper = WrapperNN(**params)


if __name__ == "__main__":
    pass
