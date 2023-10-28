from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from pathlib import Path
import os
import pickle

def highlight_max(row):
    max_value = max(row)
    bold_max_value = f"\\textbf{{{max_value}}}"
    return [bold_max_value if x == max_value else x for x in row]

# formatted_matrix = np.apply_along_axis(highlight_max, axis=1, arr=matrix)


def get_table(results, metric_name):
    # Prepare data
    data = []
    info = {}

    tissue_names = list(results['datasets'].keys())
    tissue_names = sorted(tissue_names)

    model_names = results['datasets'][tissue_names[0]]['model_results'].keys()
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
                median = round(res_tissue[mn]['scores'][metric_name]['median'], 4)
                res_model.append(median)
        data.append(res_model)

    # get table list
    data = np.array(data, dtype='object')
    # TODO add bold
    data = np.apply_along_axis(highlight_max, axis=1, arr=data)
    print(data.shape)
    tissue_names = np.array(tissue_names)
    table = np.insert(data, 0, tissue_names, axis=1)
    return table, model_names


# TODO: table
# dataset model 
def table_code(table, headers):
    """
    Code for overleaf
    """
    # table = [["Alice", 24], ["Bob", 19], ["Charlie", 32]]
    print(tabulate(table, headers=headers, numalign="center", tablefmt="latex"))


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

    # path_res = os.path.join(parent_path, "results/")
    # fns = ['scanvi_bcm_global_local_path-eval.pkl']
    print(fns)

    # ece
    for fn in fns:
        # if 'path-eval' in fn  or 'global' in fn:
            # continue

        with open(path_res + f'/{fn}', 'rb') as fh:
            results = pickle.load(fh)
        metric_name = 'F1_SCORE_MACRO'.lower()
        
        table, model_names = get_table(results, metric_name)
        header = ['Tissue']
        header.extend(model_names)
        table_code(table, header)
        break