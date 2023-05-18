from sklearn.model_selection import StratifiedKFold,LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
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

from sklearn.metrics import accuracy_score
from statistics import mean

# TODO: refactor to dataloader module
# TODO: Randomized searchCV
# TODO: outer metrics, rejection option , update res dict
# TODO: documentation
# TODO: dump predictions

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
        res = {}
        for classifier in self.classifiers:
            # Set dim_in/out for neuralNet
            if classifier.data_shape_required:
                classifier.model.set_params(module__dim_in=num_feature, module__dim_out=num_class)
            
            # Define pipeline and param_grid
            param_grid = {}
            for key, value in classifier.tuning_space.items():
                param_grid[classifier.name + '__' + key] = value

            if not classifier.preprocessing_steps:
                pipeline = Pipeline([(classifier.name, classifier.model)])
            else:
                pipeline = Pipeline(classifier.preprocessing_steps + [(classifier.name, classifier.model)])
                
                if classifier.preprocessing_params:
                    param_grid.update(classifier.preprocessing_params)

            if self.tuning_mode.lower() == 'sample':
                param_grid = self.__sample_tuning_space(param_grid)

            # Nested CV: Perform grid search with outer(model selection) and inner(parameter tuning) cross-validation
            grid_search = GridSearchCV(pipeline, param_grid, cv=KFold, scoring=inner_metrics, refit=True, n_jobs=-1)
            cv_results = cross_validate(grid_search, X, y, cv=KFold, scoring=outer_metrics, return_estimator=True, n_jobs=-1) # considering preprocessing steps

            best_params = [gs.best_params_ for gs in cv_results["estimator"]]
            best_params_unique = [str(dict(y)) for y in set(tuple(x.items()) for x in best_params)]

            logger.write(f'{classifier.name}:', msg_type='subtitle')
            logger.write(
                f'Best hyperparameters ({len(best_params_unique)}/{len(best_params)}): {", ".join(best_params_unique)}',
                msg_type='content'
            )
            temp = {}
            for score_name in outer_metrics:
                scores = cv_results[f'test_{score_name}']
                mean_score = np.mean(scores) 
                median_score = np.median(scores)
                temp[score_name] = {'mean': mean_score, 'median': median_score, 'full': scores.tolist()}

                logger.write(
                    f'{score_name.upper()}: Best Mean Score: {mean_score:.4f}  Best Median Score: {median_score:.4f}',
                    msg_type='content'
                )
            res[classifier.name] = temp
             
            logger.write(
            '',
            msg_type='content'
        )

        logger.write(
            '~~~TASK COMPLETED~~~\n\n',
            msg_type='subtitle'
        )
        return res
    

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
        
        results = self.__nested_cv(X, y, stratifiedKFold, inner_metrics, outer_metrics)

        info_dict = {}
        info_dict['random_state'] = random_seed
        info_dict['model_result'] = results
        info_dict['description'] = description
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
        selected_models=['LinearSVM'], 
        data_path=path_bgee, 
        inner_metrics='accuracy',
        outer_metrics=['accuracy', 'f1_macro'],
        task_name='testing_run'
    )
    app.run(**params)