import gdown
import zipfile
import os
import shutil


folder_id = {
    "embeddings": "1zBQmWU-yTmLVWCjmy7Ov0E-TKJTWty2D",
    "raw_data_per_tissue": "1N7FJObP4aKtabsOpgX_OIMSwpj_e95u3"
} 


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


# download_folder("embeddings")
# download_folder("raw_data_per_tissue")
