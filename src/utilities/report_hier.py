from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from pathlib import Path
import os
import pickle
from functools import partial


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

    code = tabulate(table, headers=header, colalign=colalign, tablefmt="latex_raw", floatfmt=".4f")
    return code



def get_scores(results, metric_name, model_names='all'):
    # Prepare data
    data = []
    info = {}

    tissue_names = list(results['datasets'].keys())

    try:
        tissue_names.remove('gut')
    except:
        pass

    tissue_names = sorted(tissue_names)

    if model_names=='all':
        model_names = results['datasets'][tissue_names[0]]['model_results'].keys()
        model_names = [mn for mn in model_names if mn != 'NaiveBayes']
        model_names = sorted(model_names)

    info['labels'] = list(model_names)
    info['metric_name'] = metric_name

    # print(info['labels'])
    for tn in tissue_names:
        res_tissue = results['datasets'][tn]['model_results']
        res_model =[]
        for mn in model_names:
            #TODO: refactor benchmark
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
    path_res_hier = os.path.join(parent_path, 'results/hier')

    # fns = load_res(path_res)
    fns = ['scanvi_bcm_flat.pkl']
    fns_hier = ['scanvi_bcm_global_local.pkl']

    metric_name = 'F1_SCORE_MACRO'.lower()
    best = 'max'
    
    metric_name = 'ECE'.lower()
    best = 'min'

    print(fns)

    # ================================= flat 
    model_names = ['RBFSVM', 'LogisticRegression', 'NeuralNet']
    model_names_hier = ['C-HMCNN', 'ConditionalSigmoid', 'CascadedLRPost', 'IsotonicRegressionPost']
    if not True:
        for fn in fns:
            # if 'path-eval' in fn  or 'global' in fn:
                # continue
            with open(path_res + f'/{fn}', 'rb') as fh:
                results = pickle.load(fh)
            
            data, tissue_names, model_names = get_scores(results, metric_name, model_names)

        for fn in fns_hier:
            # if 'path-eval' in fn  or 'global' in fn:
                # continue
            with open(path_res_hier + f'/{fn}', 'rb') as fh:
                results = pickle.load(fh)
            
            data_hier, tissue_names, model_names_hier = get_scores(results, metric_name, model_names_hier)

        data = np.append(data_hier, data, axis=1)
        print(data.shape)

        
        col_names = ['C-HMCNN', 'ConditionalSigmoid', 'CLR', 'IR', 'RBFSVM', 'LR', 'NeuralNet']
        # col_names = np.append(col_names, model_names, axis=0)


        latex_code = table_flat(data, col_names, tissue_names, best, header=['Tissues'], colalign=("center",) * 8)

        print("\\begin{table}[h]")
        print("\\centering")
        
        print(latex_code)

        print(f"\\caption{{F1 score macro of all tissues on scanvi\_bcm for flat, global and local models}}")
        print("\\end{table}")

    # ================================= global
    fns_hier = ['scanvi_bcm_global_local_path-eval.pkl']
    model_names_hier = ['C-HMCNN', 'ConditionalSigmoid', 'CascadedLRPost', 'IsotonicRegressionPost']

    metric_name = 'F1_SCORE_MACRO'.lower()
    best = 'max'
    
    metric_name = 'ECE'.lower()
    best = 'min'
    if  True:
        mean_data = []
        row_names = []
        col_names = []
        for fn in fns_hier:
            # if 'path-eval' in fn  or 'global' in fn:
                # continue
            with open(path_res_hier + f'/{fn}', 'rb') as fh:
                results = pickle.load(fh)
            
            data_hier, tissue_names, model_names_hier = get_scores(results, metric_name, model_names_hier)

        col_names = ['C-HMCNN', 'ConditionalSigmoid', 'CLR', 'IR']

        latex_code = table_flat(data_hier, col_names, tissue_names, best, header=['Tissues'], colalign=("center",) * 5)

        print("\\begin{table}[h]")
        print("\\centering")
        
        print(latex_code)

        print(f"\\caption{{Mean value of F1 score macro of all tissues per pre-processing method}}")
        print("\\end{table}")