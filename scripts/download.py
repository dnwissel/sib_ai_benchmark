import gdown
import zipfile
import os
import shutil


def main():

    folder_id = {
        "embeddings": "1zBQmWU-yTmLVWCjmy7Ov0E-TKJTWty2D",
        "raw_data_per_tissue": "1N7FJObP4aKtabsOpgX_OIMSwpj_e95u3"
    } 
    file_id = {
        "scanvi_bcm_head": "1fr-qgtJbN2_HCs9kZt5RulrOF6ZzXzS3",
        "scanvi_bcm_body": "1LolSyBQR4b3PG8nbIrekffhCqoGHkflR",
        "scanvi_b_head_body": "1EnN-AqGk-xdNANLIo9Rrv5emvzSRdJf5",
        "scanvi_head_body": "1wbhV5OLZZlYEL23AtILUUHxPys2WU11p",
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


    def download_file(fn):
        id = file_id[fn]
        gdown.download(id=id, quiet=False, use_cookies=False, output=f"data-raw/{fn}.zip")
        with zipfile.ZipFile(f"data-raw/{fn}.zip") as file:
            file.extractall(f"data-raw/{fn}")
        os.remove(f"data-raw/{fn}.zip")


    # download_folder("embeddings")
    # download_folder("raw_data_per_tissue")

    for fn in file_id.keys():
        download_file(fn)
    

if __name__ == "__main__":
    main()