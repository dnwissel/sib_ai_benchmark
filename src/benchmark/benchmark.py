from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, KFold, StratifiedGroupKFold, GroupKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import OrdinalEncoder

import random
import anndata
from models.wrapper import WrapperHier, WrapperHierCS, WrapperLocal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
from functools import partial
import os
import pkgutil
import importlib
from multiprocessing import Pool

from utilities.logger import Logger
from utilities.plot import plot
import logging
import json

from config import cfg
from models import flatModels, globalModels
from metrics.calibration_error import calibration_error
from scipy.special import softmax
from calibration.calibrate_model import CalibratedClassifier
# import models

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, make_scorer
from statistics import mean

from metrics.hier import f1_hier, precision_hier, recall_hier
# TODO: refactor to dataloader module
# TODO: Enable pass dataset matrix  to app 
# TODO: check if train and test have the sampe y.nunique()
# TODO: outer metrics, rejection option , update res dict
# TODO: documentation
# TODO: dump Results to disk every three(interval) classifiers in case of training failure


class Benchmark:
    def __init__(self, classifiers, datasets, tuning_mode='sample'):
        self._validate_input(tuning_mode)
        self.tuning_mode = tuning_mode
        self.classifiers = classifiers
        self.results = None
        self.datasets = datasets
        self.task_name = None
        self.path_eval = False


    def _validate_input(self,tuning_mode):
        tuning_mode_category = ['full', 'sample']
        if tuning_mode.lower() not in tuning_mode_category:
            raise ValueError(f'Available modes are: {", ".join(tuning_mode_category)}')

    def _train_single_outer_split(self):
        pass

    def _train(self, inner_cv, inner_metrics, outer_metrics, outer_cv=None, dataset=None, pre_splits=None):
        true_labels_test = []
        test_row_ids = []
        res = {}
        for classifier in self.classifiers:
            self.logger.write(f'{classifier.name}:', msg_type='subtitle')
            best_params = []
            model_result = {}
            params_search_required = True
            pipeline_steps=[]

            if pre_splits is None:
                X, y, outer_groups, row_ids = dataset
                n_splits = outer_cv.get_n_splits(X, y, outer_groups) #TODO: take care of cv method except LeaveOneGroupOut
                splits = outer_cv.split(X, y, outer_groups)
            else:
                n_splits = len(pre_splits)
                splits = pre_splits
            
            # with Pool(processes=15) as pool:
            # with Pool() as pool:
            #     pass

            # Nested CV: Perform grid search with outer(model selection) and inner(parameter tuning) cross-validation
            for fold_idx, (train, test) in enumerate(splits):
                
                if pre_splits is None:
                    X_train, X_test = X[train].astype(np.float32), X[test].astype(np.float32)
                    y_train, y_test = y[train], y[test]
                    inner_groups = outer_groups[train]
                    row_ids_split = row_ids[test]
                else:
                    X_train, X_test = train[0].astype(np.float32), test[0].astype(np.float32)
                    y_train, y_test = train[1], test[1]
                    inner_groups = train[2]
                    row_ids_split = test[3]
                    # print(X_train.shape, inner_groups.shape)

                # Initialise model
                pipeline, param_grid, y_train, y_test = classifier.init_model(X_train, y_train, y_test)

                # Hold-out validation set for calibration
                cal_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5) # TODO: Global seed
                train_idx, val_idx_cal = next(cal_cv.split(X_train, y_train)) 
                X_train, X_val_cal = X_train[train_idx], X_train[val_idx_cal]
                y_train, y_val_cal = y_train[train_idx], y_train[val_idx_cal]
                inner_groups = inner_groups[train_idx]

                if len(true_labels_test) < n_splits:
                    true_labels_test.append(y_test.tolist())
                    test_row_ids.append(row_ids_split.tolist())

                # Set Hier metric
                if self.path_eval: #TODO: refactor to general case
                    f1_hier_ = partial(f1_hier, en=classifier.encoder)
                    recall_hier_ = partial(recall_hier, en=classifier.encoder)
                    precision_hier_ = partial(precision_hier, en=classifier.encoder)

                    inner_metrics = make_scorer(f1_hier_)

                    try:
                        # print( pipeline.steps[-1][1].predict_path)
                        # pipeline.steps[-1][1].predict_path = True
                        # pipeline.steps[-1][1].set_predictPath(True)
                        classifier.model.set_predictPath(True)
                    except:
                        print("An exception occurred, set_predictPath failed")

                    outer_metrics = {'f1_hier': f1_hier_, 'recall_hier': recall_hier_, 'precision_hier': precision_hier_}
                # Fine-tuned model 
                if not param_grid:
                    model_selected = pipeline
                    params_search_required = False
                    model_selected.fit(X_train, y_train)

                # Tune Params
                else:
                    if self.tuning_mode.lower() == 'sample':
                        model_selected = RandomizedSearchCV(
                            pipeline, 
                            param_grid, 
                            cv=inner_cv, 
                            scoring=inner_metrics, 
                            n_iter=1 if cfg.debug else 15, 
                            refit=True, 
                            random_state=15, 
                            n_jobs=-1
                        ) 
                    else:
                        model_selected = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring=inner_metrics, refit=True, n_jobs=-1)
                    group_required = isinstance(inner_cv, LeaveOneGroupOut) or isinstance(inner_cv, GroupKFold) or isinstance(inner_cv, StratifiedGroupKFold)
                    model_selected.fit(X_train, y_train, groups=inner_groups if group_required else None)
                
                if params_search_required:
                    best_params.append(model_selected.best_params_)
                    if fold_idx == n_splits - 1:
                        pipeline_steps = model_selected.best_estimator_.get_params()["steps"]
                else:
                    if fold_idx == n_splits - 1:
                        # best_params = None
                        if hasattr(model_selected, 'get_params'):
                            pipeline_steps = model_selected.get_params()['steps']

                
                # for param, value in  model_selected.best_estimator_.get_params().items():
                #     print(f"{param}: {value}")
                # Log results in the last fold
                if fold_idx == n_splits - 1: 
                    if not params_search_required:
                        self.logger.write(
                            (f'Steps in pipline: {dict(pipeline_steps)}\n' # TODO: case: no pipeline 
                            f'Best hyperparameters: Not available, parameters are defined by user.'), 
                            msg_type='content'
                        )
                    else:
                        best_params_unique = [str(dict(y)) for y in set(tuple(x.items()) for x in best_params)]
                        self.logger.write(
                            (f'Steps in pipline: {dict(pipeline_steps)}\n'
                            f'Best hyperparameters ({len(best_params_unique)}/{n_splits}): {", ".join(best_params_unique)}'),
                            msg_type='content'
                        )
                # classifier.model.set_predictPath(False)
                if not self.path_eval:
                    try:
                        model_selected.best_estimator_.steps[-1][1].set_predictPath(False)
                    except:
                        pass

                y_test_predict_uncalib = model_selected.predict(X_test)
                # print(y_test_predict_uncalib)
                classifier.set_modelFitted(model_selected)
                # Uncaliberated confidence
                y_test_proba_uncalib_all, logits = classifier.predict_proba(model_selected, X_test)
                if y_test_proba_uncalib_all is not None:
                    y_test_proba_uncalib_all = y_test_proba_uncalib_all.astype(float) 

                # Caliberation: skip Probabilistic model
                y_test_proba_calib = []
                y_test_proba_uncalib = []
                y_test_predict_calib = []
                ece = []
                if isinstance(classifier, WrapperHier) or isinstance(classifier, WrapperLocal) or isinstance(classifier, WrapperHierCS):
                    pass
                else:
                    if logits is not None:
                        # model_calibrated = CalibratedClassifierCV(model_selected, cv='prefit', method="sigmoid", n_jobs=-1)
                        model_calibrated = CalibratedClassifier(classifier)
                        model_calibrated.fit(X_val_cal, y_val_cal)

                        y_test_predict_calib = model_calibrated.predict(X_test)
                        y_test_proba_calib_all = model_calibrated.predict_proba(X_test).astype(float)
                        for sample_idx, class_idx in enumerate(y_test_predict_calib):
                            # print(y_test_proba_calib_all)
                            # print(sample_idx, class_idx)
                            y_test_proba_calib.append(y_test_proba_calib_all[sample_idx, class_idx])
                            if y_test_proba_uncalib_all is not None:
                                y_test_proba_uncalib.append(y_test_proba_uncalib_all[sample_idx, class_idx])
                        y_test_predict_calib = y_test_predict_calib.tolist()
                        ece = calibration_error(y_test, y_test_predict_calib, y_test_proba_calib)
                    else: #TODO reafactor to func
                        for sample_idx, class_idx in enumerate(y_test_predict_uncalib):
                            if y_test_proba_uncalib_all is not None:
                                y_test_proba_uncalib.append(y_test_proba_uncalib_all[sample_idx, class_idx])
                        # y_test_predict_calib = y_test_predict_calib.tolist()
                        ece = calibration_error(y_test, y_test_predict_uncalib, y_test_proba_uncalib)

                model_result.setdefault('predicts_calib', []).append(y_test_predict_calib) 
                model_result.setdefault('predicts_uncalib', []).append(y_test_predict_uncalib.tolist()) 
                model_result.setdefault('proba_calib', []).append(y_test_proba_calib) 
                model_result.setdefault('proba_uncalib', []).append(y_test_proba_uncalib) 
                model_result.setdefault('logits', []).append(logits.tolist() if logits is not None else logits) 
                model_result.setdefault('ece', []).append(ece) 

                y_test_predict = y_test_predict_uncalib 

                # Calculate metrics
                for metric_name, metric in outer_metrics.items():
                    score = metric(y_test, y_test_predict)
                    scores = model_result.setdefault('scores', {})
                    scores.setdefault(f'{metric_name}', {}).setdefault('full', []).append(score)

                    if fold_idx == n_splits - 1:
                        scores_ = scores[metric_name]['full']
                        mean_score = np.mean(scores_) 
                        median_score = np.median(scores_)
                        scores[metric_name].update({'mean': mean_score, 'median': median_score})

                        self.logger.write(
                            f'{metric_name.upper()}: Best Mean Score: {mean_score:.4f}  Best Median Score: {median_score:.4f}',
                            msg_type='content'
                        )
                        res[classifier.name] = model_result
                # break
            
            self.logger.write('', msg_type='content')

        return res, true_labels_test, test_row_ids
    

    def save(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

        # with open(os.path.join(dir, self.task_name), 'w') as file:
        #     json.dump(self.results, file, indent=3, separators=(', ', ': '))
        with open(os.path.join(dir, self.task_name + '.pkl'), 'wb') as fh: 
            pickle.dump(self.results, fh, pickle.HIGHEST_PROTOCOL)
    
    
    def plot(self, dir, metric_name='f1_score_macro'):
        if self.path_eval:
            metric_name = 'f1_hier'
        plot(self.results, metric_name, os.path.join(dir, self.task_name))


    def run(self, inner_metrics, outer_metrics, task_name = 'testing_run', random_seed=15, description='', is_pre_splits=True, is_outer_cv=False, path_eval=False):
        self.task_name = task_name
        self.path_eval = path_eval
        # create logger
        self.logger = Logger(name=task_name, log_to_file=True, log_to_console=False)

        self.logger.write(f'Task {task_name.upper()} Started.', msg_type='title')
        self.logger.write(
            f'{len(self.classifiers)} model(s) loaded: {", ".join(c.name for c in self.classifiers )}', msg_type='subtitle'
            )
        self.logger.write(
            f'{len(self.datasets)} dataset(s) loaded: {", ".join(list(self.datasets.keys()) )}', msg_type='subtitle'
        )

        self.results = dict(random_state=random_seed, description=description)
        # Loop over different datasets
        # for name, path in data_paths.items():
        for dn, dataset in self.datasets.items():
            self.logger.write(f'Start benchmarking models on dataset {dn.upper()}.', msg_type='subtitle')
            # inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
            # inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            outer_cv = LeaveOneGroupOut()

            if dn in ['body', 'head']:
                inner_cv = StratifiedGroupKFold(n_splits=4)
                outer_cv = StratifiedGroupKFold(n_splits=5)
                # inner_cv = GroupKFold(n_splits=4)
                # print(dn)
            else:
                inner_cv = LeaveOneGroupOut()
                
            if not is_pre_splits:
                model_results, true_labels_test, test_row_ids = self._train(inner_cv, inner_metrics, outer_metrics, outer_cv=outer_cv, dataset=dataset)
            else:
                model_results, true_labels_test, test_row_ids = self._train(inner_cv, inner_metrics, outer_metrics, outer_cv=outer_cv,pre_splits=dataset)

            self.results.setdefault('datasets', {}).update({dn: {
                'model_results': model_results, 
                'true_labels_test': true_labels_test,
                'test_row_ids': test_row_ids
            }})
            

        self.logger.write(
            '~~~TASK COMPLETED~~~\n\n',
            msg_type='subtitle'
        )
            

if __name__ == "__main__":
    pass