from tabulate import tabulate
import numpy as np
from pathlib import Path
import os
import pickle
from functools import partial
from sklearn.metrics import f1_score


def highlight(row, type='max'):
    if type == 'max':
        target = max(row)
    else:
        row = [np.median(x) for x in row]
        target = min(row)

    # print(target)
    value_format = "{:.4f}".format(target)
    bold_value = f"\\textbf{{{value_format}}}"

    for idx, val in enumerate(row):
        if val == target:
            row[idx] = bold_value
        else:
            value_format = "{:.4f}".format(val)
            row[idx] = value_format
    # print(row)
    return row


def table_flat(data, col_names, row_names, best, header=['Tissue'], colalign=("center",) * 6):
    # data = data.astype('object')
    data = np.apply_along_axis(partial(highlight, type=best), axis=1, arr=data)
    # data = [highlight(row, type=best) for row in data]
    # data = np.array(data)
    # print(data)

    table = np.insert(data, 0, row_names, axis=1)
    header.extend(col_names)

    code = tabulate(table, headers=header, colalign=colalign,
                    tablefmt="latex_raw", floatfmt=".4f")
    return code


def get_scores(results, metric_name):
    # Prepare data
    data = []
    info = {}

    tissue_names = list(results['datasets'].keys())

    try:
        tissue_names.remove('gut')
    except:
        pass

    tissue_names = sorted(tissue_names)

    model_names = results['datasets'][tissue_names[0]]['model_results'].keys()
    model_names = [mn for mn in model_names if mn != 'NaiveBayes']
    model_names = sorted(model_names)

    info['labels'] = list(model_names)
    info['metric_name'] = metric_name

    # print(info['labels'])
    for tn in tissue_names:
        res_tissue = results['datasets'][tn]['model_results']
        res_model = []
        for mn in model_names:
            # TODO: refactor benchmark
            if metric_name in ['ece', 'ece_uc']:
                res_model.append(res_tissue[mn][metric_name])
            else:
                median = res_tissue[mn]['scores'][metric_name]['median']
                median = round(median, 4)
                res_model.append(median)
        data.append(res_model)

    # get table list
    data = np.array(data, dtype='object')

    idx = tissue_names.index('proboscis_and_maxpalp')
    tissue_names[idx] = 'proboscis\_and\_maxpalp'
    tissue_names = np.array(tissue_names)

    row_names = tissue_names
    col_names = model_names
    return data, row_names, col_names


def get_scores_uncalib(results, metric_name):
    # Prepare data
    data = []
    info = {}

    tissue_names = list(results['datasets'].keys())

    try:
        tissue_names.remove('gut')
    except:
        pass

    tissue_names = sorted(tissue_names)

    model_names = results['datasets'][tissue_names[0]]['model_results'].keys()
    model_names = [mn for mn in model_names if mn != 'NaiveBayes']
    model_names = sorted(model_names)

    info['labels'] = list(model_names)
    info['metric_name'] = metric_name

    # print(info['labels'])
    for tn in tissue_names:
        res_tissue = results['datasets'][tn]['model_results']
        true_labels = results['datasets'][tn]['true_labels_test']

        res_model = []
        for mn in model_names:
            # TODO: refactor benchmark
            if metric_name in ['ece', 'ece_uc']:
                res_model.append(res_tissue[mn][metric_name])
            else:
                preds = res_tissue[mn]['predicts_uncalib']
                # n_splits = preds.shape[0]
                n_splits = len(preds)
                scores = []
                for idx in range(n_splits):
                    score_ = f1_score(
                        true_labels[idx], preds[idx], average='macro')
                    scores.append(score_)
                median = np.median(scores)
                median = round(median, 4)
                res_model.append(median)
        data.append(res_model)

    # get table list
    data = np.array(data, dtype='object')

    idx = tissue_names.index('proboscis_and_maxpalp')
    tissue_names[idx] = 'proboscis\_and\_maxpalp'
    tissue_names = np.array(tissue_names)

    row_names = tissue_names
    col_names = model_names
    return data, row_names, col_names


# TODO: ACC against ece
def load_res(path):
    fns = []
    for fn in os.listdir(path):
        if 'pkl' in fn:
            fns.append(fn)
    return fns


if __name__ == "__main__":
    parent_path = Path(__file__).parents[2]
    path_res = os.path.join(parent_path, 'results/flat')

    fns = load_res(path_res)
    fns = ['pca__flat.pkl', 'scanvi__flat.pkl',
           'scanvi_b_flat.pkl', 'scanvi_bcm_flat.pkl']
    fns = ['scanvi_bcm_flat.pkl']

    metric_name = 'F1_SCORE_MACRO'.lower()
    best = 'max'

    # metric_name = 'ECE'.lower()
    # best = 'min'

    print(fns)

    # ================================= flat
    if True:
        print("\\begin{table}[h]")
        print("\\centering")

        for fn in fns:
            # if 'path-eval' in fn  or 'global' in fn:
            # continue
            print('\n')
            with open(path_res + f'/{fn}', 'rb') as fh:
                results = pickle.load(fh)

            # data, tissue_names, model_names = get_scores(results, metric_name)
            data, tissue_names, model_names = get_scores_uncalib(
                results, metric_name=None)
            latex_code = table_flat(data, model_names, tissue_names, best)

            print(latex_code)

            fn = fn[:-4]
            fn = fn.replace('_', '\_')
            print(f"\\caption{{F1 score macro of {fn}}}")
            print("\\hspace{3em}")

        print("\\end{table}")

    # ================================= flat mean
    if False:
        mean_data = []
        row_names = []
        col_names = []
        for fn in fns:
            # if 'path-eval' in fn  or 'global' in fn:
            # continue
            print('\n')
            with open(path_res + f'/{fn}', 'rb') as fh:
                results = pickle.load(fh)

            data, tissue_names, model_names = get_scores(results, metric_name)
            row_names = model_names
            mean = np.mean(data, axis=0)
            mean_data.append(mean)

            fn = fn[:-4]
            fn = fn.replace('_', '\_')
            col_names.append(fn)

        mean_data = np.array(mean_data)
        mean_data = mean_data.T
        # print(mean_data.shape)

        latex_code = table_flat(mean_data, col_names, row_names, best, header=[
                                'Models'], colalign=("center",) * 5)

        print("\\begin{table}[h]")
        print("\\centering")

        print(latex_code)

        print(
            f"\\caption{{Mean value of F1 score macro of all tissues per pre-processing method}}")
        print("\\end{table}")
