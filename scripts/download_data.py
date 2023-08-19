import gdown
import zipfile
import os
import shutil

embeddings_id = {
    "_bcm": "1RjAQDm3nhnQ2gJokus_VQ1furic3ro7F", # scanvi_bcm
    "_b": "1VGdoBb4saxxc2xExl3NDBSVj6tmV7Fit", # scanvi_b
    "_":"1y6ekQoCyQUSmeBglNOhb9L84l5oBEETR", # scanvi_none
}

raw_data_folder_id = {
    "raw_data_per_tissue": "1N7FJObP4aKtabsOpgX_OIMSwpj_e95u3"
} 

single_files_id = {
    "data_unionized.h5ad" : "1ovtRz8tjbVOXKTUc2K-iMzVtJm2RbRSL",
    "sib_cell_type_hierarchy.tsv" : "1dbjrQhJKfst_xe-VTDMO7Bae8re9plh8",
}

def download_embeddings(fn):
    id = embeddings_id[fn]
    output = f"data-raw/embeddings{fn}.zip"
    gdown.download(id=id, output=output, quiet=True)
    with zipfile.ZipFile(output) as file:
        file.extractall(f"data-raw/embeddings{fn}")
    os.remove(output)
    

def download_raw_data(fn):
    id = raw_data_folder_id[fn]
    gdown.download_folder(id=id, quiet=True, use_cookies=False)
    shutil.move(f'{fn}', f'data-raw/{fn}')
    for item in os.listdir(f"data-raw/{fn}"):
        if item.endswith(".zip"):
            output = f"data-raw/{fn}/{item}"
            with zipfile.ZipFile(output) as file:
                file.extractall(f"data-raw/{fn}")
            os.remove(output)
    
def download_single_file(fn):
    id = single_files_id[fn]
    output = f"data-raw/{fn}"
    gdown.download(id=id, output=output, quiet=True)
    

# for f in ["_bcm", "_b", "_"]:
    # download_embeddings(f)

# download_raw_data("raw_data_per_tissue")
# download_single_file("sib_cell_type_hierarchy.tsv")
