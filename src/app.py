import argparse
import random
from functools import partial

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             make_scorer)
from sklearn.preprocessing import MinMaxScaler

from benchmark.benchmark import Benchmark  # TODO Better importing schedule
from config import cfg
from utilities import dataLoader as dl

# TODO: Enable pass dataset matrix  to app
# TODO: check if train and test have the sampe y.nunique()
# TODO: outer metrics, rejection option , update res dict
# TODO: documentation
# TODO: dump Results to disk every three(interval) classifiers in case of training failure
# TODO: pass list to model type


def main():
    random.seed(5)

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
        '-r',
        "--deselected_models",
        nargs='+',
        type=str,
        default=None,
        help="Exclude models from running; model names, separate by space"
    )

    parser.add_argument(
        '-d',
        "--debug",
        action='store_true',
        help="Debug mode"
    )

    parser.add_argument(
        '-p',
        "--path_eval",
        action='store_true',
        help="Evaluate Path"
    )

    parser.add_argument(
        '-x',
        "--extra",
        default=None,
        help="Evaluate Path"
    )

    args = parser.parse_args()

    # Load data
    exp_name = args.experiment_name
    deselected_models = args.deselected_models
    # cfg.debug = args.debug #TODO
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

    model_type = '_'.join(model_type) if isinstance(
        model_type, list) else model_type
    task_name = exp_name + '_' + model_type
    if deselected_models is not None:
        task_name = task_name + '_wo_' + '_'.join(deselected_models)

    if args.path_eval:
        task_name = task_name + '_path-eval'

    if args.extra:
        task_name = task_name + '_' + args.extra

    f1_score_macro = partial(f1_score, average='macro')
    f1_score_weighted = partial(f1_score, average='weighted')

    params = dict(
        # inner_metrics='accuracy',
        inner_metrics=make_scorer(
            f1_score_macro) if not args.path_eval else 'f1_hier',
        outer_metrics={
            'accuracy': accuracy_score,
            'balanced_accuracy_score': balanced_accuracy_score,
            'f1_score_macro': f1_score_macro,
            'f1_score_weighted': f1_score_weighted
        },
        task_name=task_name,
        is_pre_splits=exp_cfg['is_pre_splits'],
        path_eval=args.path_eval
    )
    bm.run(**params)
    bm.save(cfg.path_res)  # TODO: refactor
    bm.plot(cfg.path_res)


if __name__ == "__main__":
    main()
