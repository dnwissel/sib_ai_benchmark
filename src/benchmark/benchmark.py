import os
import pickle
from functools import partial

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut,
                                     RandomizedSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold)

from config import cfg
from metrics.hier import f1_hier, precision_hier, recall_hier
from utilities.logger import Logger
from utilities.plot import plot

# TODO: refactor to dataloader module
# TODO: Enable pass dataset matrix  to app
# TODO: check if train and test have the sampe y.nunique()
# TODO: outer metrics, rejection option , update res dict
# TODO: documentation
# TODO: dump Results to disk every three(interval) classifiers in case of training failure


class Benchmark:
    def __init__(self, classifiers, datasets, tuning_mode='random'):
        self._validate_input(tuning_mode)
        self.tuning_mode = tuning_mode
        self.classifiers = classifiers
        self.datasets = datasets
        self.results = None
        self.task_name = None
        self.path_eval = False

    def _validate_input(self, tuning_mode):
        tuning_mode_category = ['grid', 'random']
        if tuning_mode.lower() not in tuning_mode_category:
            raise ValueError(
                f'Available modes are: {", ".join(tuning_mode_category)}')

    def _train_single_outer_split(self):
        pass

    def _train(self, inner_cv, inner_metrics, outer_metrics, outer_cv=None, dataset=None, pre_splits=None):
        true_labels_test = []
        true_labels_test_encoded = []
        idx_to_evals = []
        nodes_label = []
        test_row_ids = []
        res = {}
        for classifier in self.classifiers:
            self.logger.write(f'{classifier.name}:', msg_type='subtitle')
            best_params = []
            model_result = {}
            params_search_required = True
            pipeline_steps = []

            if pre_splits is None:
                X, y, outer_groups, row_ids = dataset
                # TODO: take care of cv method except LeaveOneGroupOut
                n_splits = outer_cv.get_n_splits(X, y, outer_groups)
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
                    X_train, X_test = X[train].astype(
                        np.float32), X[test].astype(np.float32)
                    y_train, y_test = y[train], y[test]
                    inner_groups = outer_groups[train]
                    row_ids_split = row_ids[test]
                else:
                    X_train, X_test = train[0].astype(
                        np.float32), test[0].astype(np.float32)
                    y_train, y_test = train[1], test[1]
                    inner_groups = train[2]
                    row_ids_split = test[3]

                # Initialise model
                pipeline, param_grid, y_train, y_test = classifier.init_model(
                    X_train, y_train, y_test)

                # Hold-out validation set for calibration
                cal_cv = StratifiedKFold(
                    n_splits=5, shuffle=True, random_state=5)  # TODO: Global seed
                train_idx, val_idx_cal = next(cal_cv.split(X_train, y_train))
                X_train, X_val_cal = X_train[train_idx], X_train[val_idx_cal]
                y_train, y_val_cal = y_train[train_idx], y_train[val_idx_cal]
                inner_groups = inner_groups[train_idx]

                if len(true_labels_test) < n_splits:
                    true_labels_test.append(y_test.tolist())
                    if classifier.encoder is not None:
                        nodes_label.append(classifier.encoder.labels_ordered)
                        true_labels_test_encoded.append(
                            classifier.encoder.transform(y_test))
                        idx_to_evals.append(classifier.encoder.idx_to_eval)
                    test_row_ids.append(row_ids_split.tolist())

                # Set Hier metric
                if self.path_eval:  # TODO: refactor to general case
                    f1_hier_ = partial(f1_hier, en=classifier.encoder)
                    recall_hier_ = partial(recall_hier, en=classifier.encoder)
                    precision_hier_ = partial(
                        precision_hier, en=classifier.encoder)

                    inner_metrics = make_scorer(f1_hier_)

                    try:
                        # pipeline.steps[-1][1].path_eval = True
                        # pipeline.steps[-1][1].set_predictPath(True)
                        classifier.model.set_predictPath(True)
                        classifier.set_predictPath(True)
                    except:
                        print("An exception occurred, set_predictPath failed")

                    outer_metrics = {
                        'f1_hier': f1_hier_, 'recall_hier': recall_hier_, 'precision_hier': precision_hier_}

                # Fine-tuned model
                if not param_grid:
                    model_selected = pipeline
                    params_search_required = False
                    model_selected.fit(X_train, y_train)

                # Tune Params
                else:
                    if self.tuning_mode.lower() == 'random':
                        model_selected = RandomizedSearchCV(
                            pipeline,
                            param_grid,
                            cv=inner_cv,
                            scoring=inner_metrics,
                            n_iter=1 if cfg.debug else 1,
                            refit=True,
                            random_state=15,
                            n_jobs=-1
                        )
                    else:
                        model_selected = GridSearchCV(
                            pipeline, param_grid, cv=inner_cv, scoring=inner_metrics, refit=True, n_jobs=-1)

                    # group_required = isinstance(inner_cv, LeaveOneGroupOut) or isinstance(inner_cv, GroupKFold) or isinstance(inner_cv, StratifiedGroupKFold)
                    # model_selected.fit(X_train, y_train, groups=inner_groups if group_required else None)
                    model_selected.fit(X_train, y_train, groups=inner_groups)

                if params_search_required:
                    best_params.append(model_selected.best_params_)
                    if fold_idx == n_splits - 1:
                        pipeline_steps = model_selected.best_estimator_.get_params()[
                            "steps"]
                else:
                    if fold_idx == n_splits - 1:
                        # best_params = None
                        if hasattr(model_selected, 'get_params'):
                            pipeline_steps = model_selected.get_params()[
                                'steps']

                # for param, value in  model_selected.best_estimator_.get_params().items():
                #     print(f"{param}: {value}")
                # Log results in the last fold
                if fold_idx == n_splits - 1:
                    if not params_search_required:
                        self.logger.write(
                            (f'Steps in pipline: {dict(pipeline_steps)}\n'  # TODO: case: no pipeline
                             f'Best hyperparameters: Not available, parameters are defined by user.'),
                            msg_type='content'
                        )
                    else:
                        best_params_unique = [str(dict(y)) for y in set(
                            tuple(x.items()) for x in best_params)]
                        self.logger.write(
                            (f'Steps in pipline: {dict(pipeline_steps)}\n'
                             f'Best hyperparameters ({len(best_params_unique)}/{n_splits}): {", ".join(best_params_unique)}'),
                            msg_type='content'
                        )
                classifier.set_modelFitted(model_selected)
                classifier.fit_calibrater(X_val_cal, y_val_cal)
                proba_calib, proba_uncalib = classifier.predict_proba(X_test)
                pred_calib, pred_uncalib = classifier.predict(X_test)

                ece = None
                ece_uc = None

                if self.path_eval:
                    y_test_encoded = classifier.encoder.transform(y_test)
                    ece_uc = classifier.ece_path(
                        y_test_encoded, pred_uncalib, proba_uncalib)  # TODO None

                    if pred_calib is not None:
                        ece = classifier.ece_path(
                            y_test_encoded, pred_calib, proba_calib)
                else:
                    ece_uc = classifier.ece(
                        y_test, pred_uncalib, proba_uncalib)

                    if pred_calib is not None:
                        ece = classifier.ece(y_test, pred_calib, proba_calib)

                if ece is None:
                    pred_calib = pred_uncalib
                    proba_calib = proba_uncalib
                    ece = ece_uc

                model_result.setdefault(
                    'predicts_calib', []).append(pred_calib)
                model_result.setdefault(
                    'predicts_uncalib', []).append(pred_uncalib)
                model_result.setdefault('proba_calib', []).append(proba_calib)
                model_result.setdefault(
                    'proba_uncalib', []).append(proba_uncalib)
                # model_result.setdefault('logits', []).append(logits.tolist() if logits is not None else logits)
                model_result.setdefault('ece', []).append(ece)
                model_result.setdefault('ece_uc', []).append(ece_uc)

                # y_test_pred = pred_uncalib
                y_test_pred = pred_calib

                # Calculate metrics
                for metric_name, metric in outer_metrics.items():
                    score = metric(y_test, y_test_pred)
                    scores = model_result.setdefault('scores', {})
                    scores.setdefault(f'{metric_name}', {}).setdefault(
                        'full', []).append(score)

                    if fold_idx == n_splits - 1:
                        scores_ = scores[metric_name]['full']
                        mean_score = np.mean(scores_)
                        median_score = np.median(scores_)
                        scores[metric_name].update(
                            {'mean': mean_score, 'median': median_score})

                        self.logger.write(
                            f'{metric_name.upper()}: Best Mean Score: {mean_score:.4f}  Best Median Score: {median_score:.4f}',
                            msg_type='content'
                        )
                        res[classifier.name] = model_result
                # break

            self.logger.write('', msg_type='content')

        return res, true_labels_test, test_row_ids, nodes_label, true_labels_test_encoded, idx_to_evals

    def save(self, dir_):
        if not os.path.exists(dir_):
            os.makedirs(dir_)

        # with open(os.path.join(dir_, self.task_name), 'w') as file:
        #     json.dump(self.results, file, indent=3, separators=(', ', ': '))
        with open(os.path.join(dir_, self.task_name + '.pkl'), 'wb') as fh:
            pickle.dump(self.results, fh, pickle.HIGHEST_PROTOCOL)

    def plot(self, dir_, metric_name='f1_score_macro'):
        if self.path_eval:
            metric_name = 'f1_hier'
        plot(self.results, metric_name, os.path.join(dir_, self.task_name))

    def run(self, inner_metrics, outer_metrics, task_name='testing_run', random_seed=15, description='', is_pre_splits=True, is_outer_cv=False, path_eval=False):
        self.task_name = task_name
        self.path_eval = path_eval
        # create logger
        self.logger = Logger(
            name=task_name, log_to_file=True, log_to_console=False)

        self.logger.write(
            f'Task {task_name.upper()} Started.', msg_type='title')
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
            self.logger.write(
                f'Start benchmarking models on dataset {dn.upper()}.', msg_type='subtitle')
            # inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
            # inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            outer_cv = LeaveOneGroupOut()

            if dn in ['body', 'head']:
                inner_cv = StratifiedGroupKFold(n_splits=4)
                outer_cv = StratifiedGroupKFold(n_splits=5)
                # inner_cv = GroupKFold(n_splits=4)
            else:
                inner_cv = LeaveOneGroupOut()

            if not is_pre_splits:
                model_results, true_labels_test, test_row_ids, nodes_label, true_labels_test_encoded, idx_to_evals = self._train(
                    inner_cv, inner_metrics, outer_metrics, outer_cv=outer_cv, dataset=dataset)
            else:
                model_results, true_labels_test, test_row_ids, nodes_label, true_labels_test_encoded, idx_to_evals = self._train(
                    inner_cv, inner_metrics, outer_metrics, outer_cv=outer_cv, pre_splits=dataset)

            self.results.setdefault('datasets', {}).update({dn: {
                'model_results': model_results,
                'true_labels_test': true_labels_test,
                'true_labels_test_encoded': true_labels_test_encoded,
                'idx_to_evals': idx_to_evals,
                'test_row_ids': test_row_ids,
                'nodes_label': nodes_label
            }})

        self.logger.write(
            '~~~TASK COMPLETED~~~\n\n',
            msg_type='subtitle'
        )


if __name__ == "__main__":
    pass
