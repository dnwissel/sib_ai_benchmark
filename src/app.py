import argparse
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, make_scorer

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

from utilities import dataLoader  as dl
from config import cfg

from models import flatModels, globalModels
from metrics.calibration_error import calibration_error
from calibration.calibrate_model import CalibratedClassifier
from benchmark.benchmark import Benchmark  #TODO Better importing schedule
from statistics import mean
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# TODO: Enable pass dataset matrix  to app 
# TODO: check if train and test have the sampe y.nunique()
# TODO: outer metrics, rejection option , update res dict
# TODO: documentation
# TODO: dump Results to disk every three(interval) classifiers in case of training failure
# TODO: pass list to model type

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e',
        "--experiment_name",
        type=str,
        default='scanvi_bcm',
        help="The dict key in cfg.experiments file"
    )

    parser.add_argument(
        '-m',
        "--model_type",
        nargs='+',
        type=str,
        default=None,
        help="Model types: flat, all, global, or model name;  separate by space"
    )

    parser.add_argument(
        '-d',
        "--deselected_models",
        nargs='+',
        type=str,
        default=None,
        help="model names, separate by space"
    )

    args = parser.parse_args()

    # Load data
    exp_name = args.experiment_name 
    deselected_models = args.deselected_models
    exp_cfg = cfg.experiments[exp_name]
    model_type = exp_cfg['model_type'] if args.model_type is None else args.model_type
    classifier_wrappers = dl.load_models(model_type, deselected_models)

    path = exp_cfg['data_path']
    dataloader = exp_cfg['dataloader']
    datasets = dataloader(path)

    # Set Pipeline
    for clsw in classifier_wrappers:
        preprocessing_steps = exp_cfg['ppSteps'][:]
        if clsw.name == 'NaiveBayes':
            preprocessing_steps[-1] = ('StandardScaler', MinMaxScaler())
        clsw.set_ppSteps(preprocessing_steps)

    # Run benchmark
    bm = Benchmark(classifiers=classifier_wrappers, datasets=datasets) 

    model_type = '-'.join(model_type) if isinstance(model_type, list) else model_type
    task_name = exp_name + '_' + model_type 
    if deselected_models is not None:
        task_name = task_name + '_wo_' + '-'.join(deselected_models)
    
    f1_score_macro = partial(f1_score, average='macro')
    f1_score_weighted = partial(f1_score, average='weighted')

    params = dict(
        # inner_metrics='accuracy',
        # inner_metrics=make_scorer(debug),
        inner_metrics=make_scorer(f1_score_macro),
        outer_metrics={
            'accuracy': accuracy_score, 
            'balanced_accuracy_score': balanced_accuracy_score, 
            'f1_score_macro': f1_score_macro, 
            'f1_score_weighted': f1_score_weighted
        },
        task_name=task_name,
        is_pre_splits=exp_cfg['is_pre_splits']
    )
    bm.run(**params)
    bm.save(cfg.path_res) #TODO: refactor
    bm.plot(cfg.path_res)
    

if __name__ == "__main__":
    main()

