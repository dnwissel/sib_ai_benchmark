import os
from pathlib import Path
from utilities.dataLoader import Dataloader
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler

debug = True

parent_path = Path(__file__).parents[2]
path_tissue_data = os.path.join(parent_path, "data-raw/raw_data_per_tissue")
path_embeddings = os.path.join(parent_path, "data-raw/embeddings")
path_union = os.path.join(parent_path, "data-raw/data_unionized.h5ad")
path_hier = os.path.join(parent_path, "data-raw/sib_cell_type_hierarchy.tsv")
path_res = os.path.join(parent_path, "results/")
path_debug = os.path.join(parent_path, "data-raw/processed")

n_dim = 30

dl = Dataloader()
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
        'data_path': path_debug if debug else path_embeddings,
        'dataloader': dl.load_embeddings,
        'is_pre_splits': True,
        'model_type': 'flat',
        'ppSteps': [('StandardScaler', StandardScaler())],
        'ppParams': None,
    }
}