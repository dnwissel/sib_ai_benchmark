from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import OrdinalEncoder

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

from models import flatModels, globalModels
from metrics.calibration_error import calibration_error
from scipy.special import softmax
from calibration.calibrate_model import CalibratedClassifier
# import models

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from statistics import mean

# TODO: refactor to dataloader module
# TODO: Enable pass dataset matrix  to app 
# TODO: check if train and test have the sampe y.nunique()
# TODO: outer metrics, rejection option , update res dict
# TODO: documentation
# TODO: dump Results to disk every three(interval) classifiers in case of training failure

# create logger
logger = Logger(name='App', log_to_file=True, log_to_console=False)

class App:
    def __init__(self, tuning_mode='sample', data_loader=None, outer_cv=False):
        self.__validate_input(tuning_mode)

        self.tuning_mode = tuning_mode
        self.data_loader = data_loader
        self.classifiers = []
        self.outer_cv = outer_cv

    def __sample_tuning_space(self, tuning_space):
        for key, value in tuning_space.items():
            # self.tuning_space[key] =random.sample(value, k=2)
            tuning_space[key] = [random.choice(value)]
        return tuning_space

    def __validate_input(self,tuning_mode):
        tuning_mode_category = ['full', 'sample']
        if tuning_mode.lower() not in tuning_mode_category:
            raise ValueError(f'Available modes are: {", ".join(tuning_mode_category)}')

    def __load_models(self, selected_models='all'):
        for module_info in pkgutil.iter_modules(flatModels.__path__):
            module = importlib.import_module('models.flatModels.' + module_info.name)
            if selected_models == 'all' or module.wrapper.name in selected_models:
                self.classifiers.append(module.wrapper)

        for module_info in pkgutil.iter_modules(globalModels.__path__):
            module = importlib.import_module('models.globalModels.' + module_info.name)
            if selected_models == 'all' or module.wrapper.name in selected_models:
                self.classifiers.append(module.wrapper)
        logger.write(
            f'{len(self.classifiers)} model(s) loaded: {", ".join(c.name for c in self.classifiers )}', msg_type='subtitle'
            )

    def __load_data(self, data_path=None):
        ann = anndata.read_h5ad(data_path)

        # X = ann.X
        # y = ann.obs['cellTypeId'].cat.codes
        X = ann.X[:100]
        y = ann.obs['cellTypeId'][:100].cat.codes
        if 'batch' in ann.obs.columns:
            groups = ann.obs['batch']
        # Datatype for torch tensors
        X, y = X.astype(float32), y.astype(np.int64)

        logger.write('Data loaded.', msg_type='subtitle')
        return X, y


    def __train(self, cv, inner_metrics, outer_metrics, X=None, y=None, splits=None):
        true_labels_test = []
        res = {}
        for classifier in self.classifiers:
            
            logger.write(f'{classifier.name}:', msg_type='subtitle')
            best_params = []
            model_result = {}
            # n_splits =cv.n_splits
            n_splits =len(splits)
            params_search_required = True
            pipeline_steps=[]
            # Nested CV: Perform grid search with outer(model selection) and inner(parameter tuning) cross-validation
            # for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            for fold_idx, (train, test) in enumerate(splits):
                # Encode labels
                # y_all = np.concatenate((train[1], test[1]), axis=0)
                le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
                y_train = le.fit_transform(train[1].to_numpy().reshape(-1, 1)).flatten()
                # bounday = train[1].shape[0]
                X_train, X_test = train[0], test[0]
                y_train, y_test = y_train, le.transform(test[1].to_numpy().reshape(-1, 1)).flatten()

                # Initialise model
                pipeline, param_grid = classifier.init_model(X_train, test[1])

                # Hold-out validation set for calibration
                train_idx, val_idx_cal = next(cv.split(X_train, y_train))
                X_train, X_val_cal = X_train[train_idx], X_train[val_idx_cal]
                y_train, y_val_cal = y_train[train_idx], y_train[val_idx_cal]

                if len(true_labels_test) < n_splits:
                    true_labels_test.append(y_test.tolist())

                # Fine-tuned model 
                if not param_grid:
                    model_selected = pipeline
                    params_search_required = False
                # Tune Params
                else:
                    if self.tuning_mode.lower() == 'sample':
                        # model_selected = RandomizedSearchCV(pipeline, param_grid, cv=cv, scoring=inner_metrics, n_iter=30, refit=True, n_jobs=-1)
                        model_selected = RandomizedSearchCV(pipeline, param_grid, cv=cv, scoring=inner_metrics, n_iter=1, refit=True, n_jobs=-1) # For debug
                    else:
                        model_selected = GridSearchCV(pipeline, param_grid, cv=cv, scoring=inner_metrics, refit=True, n_jobs=-1)
                model_selected.fit(X_train, y_train)
                
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
                        logger.write(
                            (f'Steps in pipline: {dict(pipeline_steps)}\n' # TODO: case: no pipeline 
                            f'Best hyperparameters: Not available, parameters are defined by user.'), 
                            msg_type='content'
                        )
                    else:
                        best_params_unique = [str(dict(y)) for y in set(tuple(x.items()) for x in best_params)]
                        logger.write(
                            (f'Steps in pipline: {dict(pipeline_steps)}\n'
                            f'Best hyperparameters ({len(best_params_unique)}/{n_splits}): {", ".join(best_params_unique)}'),
                            msg_type='content'
                        )
                y_test_predict_uncalib = model_selected.predict(X_test)
                classifier.set_modelFitted(model_selected)
                # Uncaliberated confidence
                y_test_proba_uncalib_all, logits = classifier.predict_proba(model_selected, X_test)
                y_test_proba_uncalib_all = y_test_proba_uncalib_all.astype(float)
                # Caliberation: skip Probabilistic model
                y_test_proba_calib = []
                y_test_proba_uncalib = []
                y_test_predict_calib = []
                if logits is not None:
                    # model_calibrated = CalibratedClassifierCV(model_selected, cv='prefit', method="sigmoid", n_jobs=-1)
                    model_calibrated = CalibratedClassifier(classifier)
                    model_calibrated.fit(X_val_cal, y_val_cal)

                    y_test_predict_calib = model_calibrated.predict(X_test)
                    y_test_proba_calib_all = model_calibrated.predict_proba(X_test).astype(float)
                    for sample_idx, class_idx in enumerate(y_test_predict_calib):
                        y_test_proba_calib.append(y_test_proba_calib_all[sample_idx, class_idx])
                        if y_test_proba_uncalib_all is not None:
                            y_test_proba_uncalib.append(y_test_proba_uncalib_all[sample_idx, class_idx])
                    y_test_predict_calib = y_test_predict_calib.tolist()
                    print(calibration_error(y_test, y_test_predict_calib, y_test_proba_calib))

                model_result.setdefault('predicts_calib', []).append(y_test_predict_calib) 
                model_result.setdefault('predicts_uncalib', []).append(y_test_predict_uncalib.tolist()) 
                model_result.setdefault('proba_calib', []).append(y_test_proba_calib) 
                model_result.setdefault('proba_uncalib', []).append(y_test_proba_uncalib) 

                y_test_predict = y_test_predict_uncalib 
                # Calculate metrics
                for metric_name, metric in outer_metrics.items():
                    score = metric(y_test_predict, y_test)
                    model_result.setdefault(f'{metric_name}', {}).setdefault('full', []).append(score)

                    if fold_idx == n_splits - 1:
                        scores = model_result[metric_name]['full']
                        mean_score = np.mean(scores) 
                        median_score = np.median(scores)
                        model_result[metric_name].update({'mean': mean_score, 'median': median_score})

                        logger.write(
                            f'{metric_name.upper()}: Best Mean Score: {mean_score:.4f}  Best Median Score: {median_score:.4f}',
                            msg_type='content'
                        )
                        res[classifier.name] = model_result
                break
            
            logger.write('', msg_type='content')

        return res, true_labels_test
    

    def __save(self, results, file_name):
        current_file_dir = os.path.dirname(__file__)
        dir = os.path.join(current_file_dir, '../results/.temp')
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(os.path.join(dir, file_name), 'w') as file:
            json.dump(results, file, indent=3, separators=(', ', ': '))


    def run(self, inner_metrics, outer_metrics, data_paths=None, datasets=None, selected_models='all', task_name = 'testing_run', random_seed=15, description=''):
        logger.write(f'Task {task_name.upper()} Started.', msg_type='title')
        self.__load_models(selected_models)

        info_dict = dict(random_state=random_seed, description=description)
        # Loop over different datasets
        # for name, path in data_paths.items():
        for name, dataset in datasets.items():
            logger.write(f'Start benchmarking models on dataset {name.upper()}.', msg_type='subtitle')
            # X, y = self.__load_data(path)  
            # X, y, groups = dataset  
            splits = dataset
            stratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
            # results, true_labels_test = self.__train(X, y, stratifiedKFold, inner_metrics, outer_metrics)
            results, true_labels_test = self.__train(stratifiedKFold, inner_metrics, outer_metrics, splits = dataset)
            info_dict.update({name: {'model_results': results, 'true_labels_test': true_labels_test}})
            
        self.__save(info_dict, task_name) # dump the predicts

        logger.write(
            '~~~TASK COMPLETED~~~\n\n',
            msg_type='subtitle'
        )
            
        
            

if __name__ == "__main__":
    # Get raw data path
    current_file_dir = os.path.dirname(__file__)
    data_dir = os.path.join(os.path.dirname(current_file_dir), 'data-raw')
    path_asap = data_dir + "/ASAP41_final.h5ad"
    path_bgee = data_dir + "/SRP200614.h5ad"

    # Load raw data
    # ann = anndata.read_h5ad(path_bgee)
    # # X = ann.X
    # # y = ann.obs['cellTypeId'].cat.codes
    # mask = ann.obs['cellTypeId'] != 'unannotated'
    # X = ann.X[mask][:100]
    # y = ann.obs[mask]['cellTypeId'].cat.codes[:100]
    
    # groups = None
    # if 'batch' in ann.obs.columns:
    #     groups = ann.obs['batch']
    # # Datatype for torch tensors
    # X, y = X.astype(float32), y.astype(np.int64)
    # bgee = (X.toarray(), y, groups)

    # Load embeddings
    path =  data_dir + '/processed'
    splits_fn = {}
    for fn in os.listdir(path):
        if '_train_' in fn or '_test_' in fn:
            tissue_name = fn[:fn.index('_')] #TODO: debug index of _train
            splits_fn.setdefault(tissue_name, []).append(fn)
    # print(len(splits_fn))
    splits_data = {}

    for k, v in splits_fn.items():
        v = sorted(v, key = lambda x : (x[x.rindex('_') + 1:], x[x.rindex('_') - 1]))
        # v = sorted(v, key = lambda x : (x[x.rindex('_') + 1], x[x.rindex('_') - 1]), reverse=True)
        splits_fn[k] = [(v[i],v[i + 1]) for i in range(0,len(v), 2)]
        data = []
        for i in range(0,len(v), 2):
            ann_train = anndata.read_h5ad(path + '/' + v[i])
            ann_test = anndata.read_h5ad(path + '/' + v[i + 1])
            splits_data.setdefault(k, []).append([(ann_train.X, ann_train.obs['y']), (ann_test.X, ann_test.obs['y'])])
    
    # Run App
    app = App(tuning_mode="sample") 
    
    params = dict(
        # selected_models=['NeuralNet'], 
        selected_models=['C-HMCNN'], 
        # data_paths={'bgee': path_bgee, 'asap': path_asap},
        datasets = splits_data,
        inner_metrics='accuracy',
        outer_metrics={'accuracy': accuracy_score, 'balanced_accuracy_score': balanced_accuracy_score, 'f1_score_macro': partial(f1_score, average='macro'), 'f1_score_weighted': partial(f1_score, average='weighted')},
        # outer_metrics={'accuracy': accuracy_score, 'f1_score': f1_score },
        task_name='testing_run'
    )
    app.run(**params)