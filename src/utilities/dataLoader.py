import importlib
import os
import pkgutil
import anndata
import numpy as np
import pandas as pd
import networkx as nx

from models import flatModels, globalModels


def load_pre_splits(path, batch_min=3, is_row_id=True):
    splits_fn = {}
    for fn in os.listdir(path):
        idx = fn.find('_train_')
        if idx == -1:
            idx = fn.find('_test_')
        if idx != -1:
            tissue_name = fn[:idx]
            splits_fn.setdefault(tissue_name, []).append(fn)

    splits_data = {}
    for k, v in splits_fn.items():
        v = sorted(v, key = lambda x : (x[x.rindex('_') + 1:], x[x.rindex('_') - 1]))
        splits_fn[k] = [(v[i],v[i + 1]) for i in range(0,len(v), 2)]

        ann_train = anndata.read_h5ad(path + '/' + v[0])
        groups_train = ann_train.obs['batch_id']
        if groups_train.nunique() < batch_min - 1:
            continue

        for i in range(0,len(v), 2):
            ann_train = anndata.read_h5ad(path + '/' + v[i])
            ann_test = anndata.read_h5ad(path + '/' + v[i + 1])
            # print(ann_train.obs.columns)
            if is_row_id:
                # print(ann_train.obs['y'].shape)
                # print(ann_train.obs['batch_id'].shape)
                splits_data.setdefault(k, []).append(
                    [(ann_train.X, ann_train.obs['y'], ann_train.obs['batch_id'], ann_train.obs['id']), 
                    (ann_test.X, ann_test.obs['y'], ann_test.obs['batch_id'], ann_test.obs['id'])]
                )
            else:
                splits_data.setdefault(k, []).append(
                    [(ann_train.X, ann_train.obs['y'], ann_train.obs['batch_id']), 
                    (ann_test.X, ann_test.obs['y'], ann_test.obs['batch_id'])]
                )

    return splits_data


def load_tissue_raw(path, batch_min=3, is_row_id=True):
    splits_fn = {}
    for fn in os.listdir(path):
        idx = fn.find('_pp')
        if idx != -1:
            tissue_name = fn[:idx]
            splits_fn.setdefault(tissue_name, []).append(fn)

    splits_data = {}
    for k, v in splits_fn.items():
        assert len(v) == 1
        adata = anndata.read_h5ad(path + '/' + v[0])
        groups = adata.obs['batch_id']
        if groups.nunique() < batch_min:
            continue
        if is_row_id:
            splits_data[k] = (adata.X, adata.obs['y'], adata.obs['batch_id'], adata.obs['id'])
        else:
            splits_data[k] = (adata.X, adata.obs['y'], adata.obs['batch_id'])
    return splits_data


def load_raw_data(path):
    ann = anndata.read_h5ad(path)
    # X = ann.X
    # y = ann.obs['cellTypeId'].cat.codes
    mask = ann.obs['cellTypeId'] != 'unannotated'
    X = ann.X[mask][:100]
    y = ann.obs[mask]['cellTypeId'].cat.codes[:100]
    
    groups = None
    if 'batch' in ann.obs.columns:
        groups = ann.obs['batch']
    # Datatype for torch tensors
    X, y = X.astype(np.float32), y.astype(np.int64)
    datasets = {'raw_data': (X.toarray(), y, groups)}
    return datasets


def load_models(selected_models='all', deselected_models=None):
    classifiers = []
    for module_info in pkgutil.iter_modules(flatModels.__path__):
        module = importlib.import_module('models.flatModels.' + module_info.name)
        if deselected_models is not None and module.wrapper.name in deselected_models:
            continue
        if 'all' in selected_models or 'flat' in selected_models  or module.wrapper.name in selected_models:
            classifiers.append(module.wrapper)

    for module_info in pkgutil.iter_modules(globalModels.__path__):
        module = importlib.import_module('models.globalModels.' + module_info.name)
        if deselected_models is not None and module.wrapper.name in deselected_models:
            continue
        if 'all' in selected_models or 'global' in selected_models  or module.wrapper.name in selected_models:
            classifiers.append(module.wrapper)
    # logger.write(
    #     f'{len(classifiers)} model(s) loaded: {", ".join(c.name for c in classifiers )}', msg_type='subtitle'
    #     )
    return classifiers


def load_full_hier(path):
    hier = pd.read_csv(path, sep='\t')
    # hier = hier.replace(':', '-', regex=True)
    edges = hier.to_records(index=False).tolist()
    # print(f"#Edges:{len(edges)}")
    G = nx.DiGraph(edges).reverse()
    roots_label = [v for v, d in G.in_degree() if d == 0]
    if len(roots_label) > 1:
        for node in roots_label:
            G.add_edge('root', node)
        roots_label.append('root')
        # Add missing labels in the Hier
        G.add_edge('root', 'FBbt:00005073')
    # print(roots_label)
    return G, roots_label

