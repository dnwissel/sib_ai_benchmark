import os
from pathlib import Path
from utilities import dataLoader  as dl
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# debug = False
debug = True

#TODO: put embeddings under data/
parent_path = Path(__file__).parents[2]
path_tissue_data = os.path.join(parent_path, "data-raw/raw_data_per_tissue")

path_scanvi_bcm = os.path.join(parent_path, "data-raw/scanvi_bcm")
path_scanvi_b = os.path.join(parent_path, "data-raw/scanvi_b")
path_scanvi_ = os.path.join(parent_path, "data-raw/scanvi_")

path_pca_ = os.path.join(parent_path, "data-raw/pca_")

path_pf_ = os.path.join(parent_path, "data-raw/pf_")

path_union = os.path.join(parent_path, "data-raw/data_unionized.h5ad")
path_hier = os.path.join(parent_path, "data-raw/sib_cell_type_hierarchy.tsv")
path_res = os.path.join(parent_path, "results/")
path_debug = os.path.join(parent_path, "data-raw/debug")

n_dim = 30

experiments = {
    'pca_': {
        'data_path': path_pca_ if not debug else path_debug,
        'dataloader': dl.load_tissue_raw,
        'is_pre_splits': False,
        'model_type': 'flat',
        'ppSteps': [('DimensionReduction', TruncatedSVD(n_components=n_dim)),('StandardScaler', StandardScaler())],
        'ppParams': None,
    },
    
    # Only do filtering
    'pf_': {
        'data_path': path_pf_,
        'dataloader': dl.load_tissue_raw,
        'is_pre_splits': False,
        'model_type': 'flat',
        'ppSteps': [('StandardScaler', StandardScaler())],
        'ppParams': None,
    },
    
    'scanvi_bcm': {
        'data_path': path_scanvi_bcm if not debug else path_debug,
        # 'data_path': path_debug,
        'dataloader': dl.load_pre_splits,
        'is_pre_splits': True,
        'model_type': 'flat',
        'ppSteps': [('StandardScaler', StandardScaler())],
        'ppParams': None,
    },

    'scanvi_b': {
        'data_path': path_debug if debug else path_scanvi_b,
        'dataloader': dl.load_pre_splits,
        'is_pre_splits': True,
        'model_type': 'flat',
        'ppSteps': [('StandardScaler', StandardScaler())],
        'ppParams': None,
    },

    'scanvi_': {
        'data_path': path_debug if debug else path_scanvi_,
        'dataloader': dl.load_pre_splits,
        'is_pre_splits': True,
        'model_type': 'flat',
        'ppSteps': [('StandardScaler', StandardScaler())],
        'ppParams': None,
    }
}