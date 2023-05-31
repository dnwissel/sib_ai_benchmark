from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import random
import anndata
import numpy as np
import pandas as pd

import os
import pkgutil
import importlib
from utilities.logger import Logger
import logging
import json

from models import flatModels
# import models

from sklearn.metrics import accuracy_score, f1_score
from statistics import mean

# TODO: refactor to dataloader module
# TODO: Randomized searchCV
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

        logger.write(
            f'{len(self.classifiers)} model(s) loaded: {", ".join(c.name for c in self.classifiers )}', msg_type='subtitle'
            )

    def __load_data(self, data_path):
        ann = anndata.read_h5ad(data_path)

        X = ann.X
        y = ann.obs['cellTypeId'].cat.codes
        # X = ann.X[:100]
        # y = ann.obs['cellTypeId'].cat.codes[:100]
        if 'batch' in ann.obs.columns:
            groups = ann.obs['batch']
        # Datatype for torch tensors
        X, y = X.astype(np.float32), y.astype(np.int64)

        logger.write('Data loaded.', msg_type='subtitle')
        return X, y


    def __nested_cv(self, X, y, KFold, inner_metrics, outer_metrics):
        num_feature, num_class = X.shape[1], y.nunique()
        true_labels_test = []
        res = {}
        for classifier in self.classifiers:
            # Set dim_in/out for neuralNet
            if classifier.data_shape_required:
                classifier.model.set_params(module__dim_in=num_feature, module__dim_out=num_class)
            
            # Define pipeline and param_grid
            param_grid = {}
            if classifier.tuning_space:
                for key, value in classifier.tuning_space.items():
                    param_grid[classifier.name + '__' + key] = value

            if not classifier.preprocessing_steps:
                pipeline = Pipeline([(classifier.name, classifier.model)])
            else:
                pipeline = Pipeline(classifier.preprocessing_steps + [(classifier.name, classifier.model)])
                
                if classifier.preprocessing_params:
                    param_grid.update(classifier.preprocessing_params)
            
            logger.write(f'{classifier.name}:', msg_type='subtitle')
            best_params = []
            model_result = {}
            n_splits =KFold.n_splits
            params_search_required = True
            pipeline_steps=[]
            # Nested CV: Perform grid search with outer(model selection) and inner(parameter tuning) cross-validation
            for fold_idx, (train_idx, test_idx) in enumerate(KFold.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if len(true_labels_test) < n_splits:
                    true_labels_test.append(y_test.tolist())

                # Fine-tuned model 
                if not param_grid:
                    model_selected = pipeline
                    params_search_required = False
                # Tune Params
                else:
                    if self.tuning_mode.lower() == 'sample':
                        model_selected = RandomizedSearchCV(pipeline, param_grid, cv=KFold, scoring=inner_metrics,n_iter=10, refit=True, n_jobs=-1)
                    else:
                        model_selected = GridSearchCV(pipeline, param_grid, cv=KFold, scoring=inner_metrics, refit=True, n_jobs=-1)
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

                y_test_predict = model_selected.predict(X_test)
                model_result.setdefault('predicts', []).append(y_test_predict.tolist()) 
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
                
            logger.write('', msg_type='content')

        logger.write(
            '~~~TASK COMPLETED~~~\n\n',
            msg_type='subtitle'
        )
        return res, true_labels_test
    

    def __dump_to_disk(self, results, file_name):
        current_file_dir = os.path.dirname(__file__)
        dir = os.path.join(current_file_dir, '../results/.temp')
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(os.path.join(dir, file_name), 'w') as file:
            json.dump(results, file, indent=3, separators=(', ', ': '))


    def run(self, data_path, inner_metrics, outer_metrics, selected_models='all', task_name = 'testing_run', random_seed=15, description=''):
        logger.write(f'Task {task_name.upper()} Started.', msg_type='title')
        X, y = self.__load_data(data_path)        
        self.__load_models(selected_models)

        stratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        
        results, true_labels_test = self.__nested_cv(X, y, stratifiedKFold, inner_metrics, outer_metrics)

        info_dict = dict(random_state=random_seed, model_result=results, description=description, true_labels_test=true_labels_test)
        
        self.__dump_to_disk(info_dict, task_name) # dump the predicts
            
        
            

if __name__ == "__main__":
    # Get raw data path
    current_file_dir = os.path.dirname(__file__)
    data_dir = os.path.join(os.path.dirname(current_file_dir), 'data-raw')
    path_asap = data_dir + "/ASAP41_final.h5ad"
    path_bgee = data_dir + "/SRP200614.h5ad"

    # Run App
    app = App(tuning_mode="sample") 
    
    params = dict(
        # selected_models=['RBFSVM'], 
        data_path=path_bgee, 
        inner_metrics='accuracy',
        outer_metrics={'accuracy': accuracy_score},
        # outer_metrics={'accuracy': accuracy_score, 'f1_score': f1_score },
        task_name='testing_run'
    )
    app.run(**params)