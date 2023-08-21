import os
from pathlib import Path
from utilities import dataLoader  as dl
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler

debug = True

parent_path = Path(__file__).parents[2]
path_tissue_data = os.path.join(parent_path, "data-raw/raw_data_per_tissue")

path_embeddings_bcm = os.path.join(parent_path, "data-raw/embeddings_bcm")
path_embeddings_b = os.path.join(parent_path, "data-raw/embeddings_b")
path_embeddings_ = os.path.join(parent_path, "data-raw/embeddings_")

path_union = os.path.join(parent_path, "data-raw/data_unionized.h5ad")
path_hier = os.path.join(parent_path, "data-raw/sib_cell_type_hierarchy.tsv")
path_res = os.path.join(parent_path, "results/")
path_debug = os.path.join(parent_path, "data-raw/processed")

n_dim = 30


experiments = {
    'pca_': {
        'data_path': path_tissue_data,
        'dataloader': dl.load_tissue_raw,
        'is_pre_splits': False,
        'model_type': 'flat',
        'ppSteps': [('DimensionReduction', TruncatedSVD(n_components=n_dim)),('StandardScaler', StandardScaler())],
        'ppParams': None,
    },
    
    'scanvi_bcm': {
        # 'data_path': path_embeddings_bcm,
        'data_path': path_debug,
        'dataloader': dl.load_embeddings,
        'is_pre_splits': True,
        'model_type': 'flat',
        'ppSteps': [('StandardScaler', StandardScaler())],
        'ppParams': None,
    },

    'scanvi_b': {
        'data_path': path_debug if debug else path_embeddings_b,
        'dataloader': dl.load_embeddings,
        'is_pre_splits': True,
        'model_type': 'flat',
        'ppSteps': [('StandardScaler', StandardScaler())],
        'ppParams': None,
    },

    'scanvi_': {
        'data_path': path_debug if debug else path_embeddings_,
        'dataloader': dl.load_embeddings,
        'is_pre_splits': True,
        'model_type': 'flat',
        'ppSteps': [('StandardScaler', StandardScaler())],
        'ppParams': None,
    }
}