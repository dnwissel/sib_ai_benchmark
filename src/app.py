from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

import random
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from functools import partial
import os
import pkgutil
import importlib
from utilities.logger import Logger
import logging
import json

from utilities.hier import Encoder, load_full_hier, get_lm, get_R
from utilities.dataLoader import Dataloader

from models import flatModels, globalModels
from metrics.calibration_error import calibration_error
from calibration.calibrate_model import CalibratedClassifier
from benchmark.benchmark import Benchmark  #TODO Better importing schedule
from statistics import mean
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# TODO: refactor to dataloader module
# TODO: Enable pass dataset matrix  to app 
# TODO: check if train and test have the sampe y.nunique()
# TODO: outer metrics, rejection option , update res dict
# TODO: documentation
# TODO: dump Results to disk every three(interval) classifiers in case of training failure


def main():
    # Data path
    current_file_dir = os.path.dirname(__file__)
    data_dir = os.path.join(os.path.dirname(current_file_dir), 'data-raw')
    # path_union = data_dir + "/data_unionized.h5ad"
    path_hier = data_dir + "/sib_cell_type_hierarchy.tsv"
    path_embedings =  data_dir + '/processed'

    # Load data
    dl = Dataloader()
    # datasets = load_raw_data(path_union)
    datasets = dl.load_embedings(path_embedings)

    # Load models
    flat_wrappers = dl.load_models('flat')
    # flat_wrappers = dl.load_models('LogisticRegression')
    global_wrappers = dl.load_models('global')
    for wrapper in global_wrappers:
        wrapper.set_gGlobal(*load_full_hier(path_hier))


    #=============== PCA ===============
    classifier_wrappers = flat_wrappers
    print(classifier_wrappers)
    # Set Pipeline
    n_dim = 30
    for clsw in classifier_wrappers:
        preprocessing_steps=[('DimensionReduction', TruncatedSVD(n_components=n_dim)),('StandardScaler', StandardScaler())]
        if clsw.name == 'NaiveBayes':
            preprocessing_steps=[('DimensionReduction', TruncatedSVD(n_components=n_dim)),('StandardScaler', MinMaxScaler())]
        clsw.set_ppSteps(preprocessing_steps)

    bm = Benchmark(classifiers=classifier_wrappers, datasets=datasets) 
    
    params = dict(
        inner_metrics='accuracy',
        outer_metrics={'accuracy': accuracy_score, 'balanced_accuracy_score': balanced_accuracy_score, 'f1_score_macro': partial(f1_score, average='macro'), 'f1_score_weighted': partial(f1_score, average='weighted')},
        task_name='testing_run'
    )
    bm.run(**params)


if __name__ == "__main__":
   main()