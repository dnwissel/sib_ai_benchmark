import gdown
import zipfile
import os
import shutil


folder_id = {
    "embeddings": "1zBQmWU-yTmLVWCjmy7Ov0E-TKJTWty2D",
    "raw_data_per_tissue": "1N7FJObP4aKtabsOpgX_OIMSwpj_e95u3"
} 

# single_file_id = {
#     "data_unionized.h5ad" : "1ovtRz8tjbVOXKTUc2K-iMzVtJm2RbRSL",
#     "sib_cell_type_hierarchy.tsv" : "1dbjrQhJKfst_xe-VTDMO7Bae8re9plh8",
#     "results.json": "1F1xyRk8F6WG1tSoJLnqVPL8PLwoDyylO",
# }


def download_folder(fn):
    id = folder_id[fn]
    gdown.download_folder(id=id, quiet=False, use_cookies=False)
    shutil.move(f'{fn}', f'data-raw/{fn}')
    for item in os.listdir(f"data-raw/{fn}"):
        if item.endswith(".zip"):
            output = f"data-raw/{fn}/{item}"
            with zipfile.ZipFile(output) as file:
                if fn == "embeddings":
                    file.extractall(f"data-raw/{item[:-4]}")
                elif fn == "raw_data_per_tissue":
                    file.extractall(f"data-raw/{fn}")
            os.remove(output)
    os.rmdir(f'data-raw/{fn}')


# def download_single_file(fn):
#     id = single_file_id[fn]
#     output = f"data-raw/{fn}"
#     gdown.download(id=id, output=output, quiet=False)


# download_folder("embeddings")
# download_folder("raw_data_per_tissue")
# download_single_file("sib_cell_type_hierarchy.tsv")
