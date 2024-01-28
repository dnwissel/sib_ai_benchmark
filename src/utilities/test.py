# Test
import os
from pathlib import Path

from utilities import dataLoader as dl

parent_path = Path(__file__).parents[2]

path_embeddings_bcm = os.path.join(parent_path, "data-raw/embeddings_bcm")
path_hier = os.path.join(parent_path, "data-raw/sib_cell_type_hierarchy.tsv")
G, _ = dl.load_full_hier(path_hier)
data = dl.load_pre_splits(path_embeddings_bcm)
for k, v in data.items():
    y = set(v[0][0][1].unique()) | set(v[0][1][1].unique())
    diff = y - G.nodes()
    if len(diff) > 0:
        print(k, diff)
