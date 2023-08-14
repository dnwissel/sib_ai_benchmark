import importlib
import os
import pkgutil
import anndata
import numpy as np
import pandas as pd
import networkx as nx

from models import flatModels, globalModels


class Dataloader:
    def __init__(self) -> None:
        pass

    def load_embedings(self, path):
        splits_fn = {}
        for fn in os.listdir(path):
            if '_train_' in fn or '_test_' in fn:
                tissue_name = fn[:fn.index('_')] #TODO: debug index of _train
                splits_fn.setdefault(tissue_name, []).append(fn)
        # print(len(splits_fn))

        splits_data = {}
        for k, v in splits_fn.items():
            v = sorted(v, key = lambda x : (x[x.rindex('_') + 1:], x[x.rindex('_') - 1]))
            splits_fn[k] = [(v[i],v[i + 1]) for i in range(0,len(v), 2)]
            for i in range(0,len(v), 2):
                ann_train = anndata.read_h5ad(path + '/' + v[i])
                ann_test = anndata.read_h5ad(path + '/' + v[i + 1])
                splits_data.setdefault(k, []).append([(ann_train.X, ann_train.obs['y']), (ann_test.X, ann_test.obs['y'])])
        return splits_data


    def load_raw_data(self, path):
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
        X, y = X.astype(float32), y.astype(np.int64)
        datasets = {'raw_data': (X.toarray(), y, groups)}
        return datasets


    def load_models(self, selected_models='all'):
        classifiers = []
        for module_info in pkgutil.iter_modules(flatModels.__path__):
            module = importlib.import_module('models.flatModels.' + module_info.name)
            if selected_models in ['all', 'flat'] or module.wrapper.name in selected_models:
                classifiers.append(module.wrapper)

        for module_info in pkgutil.iter_modules(globalModels.__path__):
            module = importlib.import_module('models.globalModels.' + module_info.name)
            if selected_models in ['all', 'global'] or module.wrapper.name in selected_models:
                classifiers.append(module.wrapper)
        # logger.write(
        #     f'{len(classifiers)} model(s) loaded: {", ".join(c.name for c in classifiers )}', msg_type='subtitle'
        #     )
        return classifiers

    def load_full_hier(self, path):
        hier = pd.read_csv(path, sep='\t')
        hier = hier.replace(':', '-', regex=True)
        edges = hier.to_records(index=False).tolist()
        # print(f"#Edges:{len(edges)}")
        G = nx.DiGraph(edges).reverse()
        roots_label = [v for v, d in G.in_degree() if d == 0]
        for node in roots_label:
            G.add_edge('root', node)
        roots_label.append('root')
        # print(roots_label)
        return G, roots_label

