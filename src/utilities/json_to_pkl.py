import json
import os
import pickle
from pathlib import Path

fn = 'file_name'
res_dir = os.path.join(Path(__file__).parents[2], 'results')
file_path = os.path.join(res_dir, fn)
with open(file_path) as fh:
    jf = json.load(fh)
    # print(jf.keys())

with open(file_path, 'wb') as fh:
    pickle.dump(jf, fh, pickle.HIGHEST_PROTOCOL)
