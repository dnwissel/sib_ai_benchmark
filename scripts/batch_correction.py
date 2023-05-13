import scvi
import scanpy as sc
import anndata as ad
import argparse
import logging
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pathlib import Path
from abc import ABC, abstractmethod

sc.settings.verbosity = 2
sc.logging.print_header()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)


class BatchCorrector(ABC):
    def __init__(self, keys):
        self.keys = keys

    @abstractmethod
    def correct_batch_effect(self, data):
        pass


class RegressOut(BatchCorrector):
    def correct_batch_effect(self, data):
        keys_num = []
        keys_to_drop = []
        for key in self.keys:
            print(f"{key}: {data.obs[key].dtype}")
            if not is_numeric_dtype(data.obs[key]):
                print(f"Factorizing {key}")
                data.obs[f"{key}_num"] = pd.factorize(data.obs[key])[0]
                new_col = data.obs[f"{key}_num"]
                print(f"{key}_num: {new_col.dtype}")
                keys_num.append(f"{key}_num")
                keys_to_drop.append(f"{key}_num")
            else:
                print(f"{key}: {data.obs[key].dtype}")
                keys_num.append(key)
        print(f"keys_num: {keys_num}")
        sc.pp.regress_out(data, keys_num)
        data.obs.drop(columns=keys_to_drop, inplace=True)


class Scanvi(BatchCorrector):
    def correct_batch_effect(self, data):
        scvi.model.SCVI.setup_anndata(data, layer="counts", batch_key=self.keys[0])
        vae = scvi.model.SCVI(data)
        vae.train()
        data.obsm["X_scVI"] = vae.get_latent_representation()
        lvae = scvi.model.SCANVI.from_scvi_model(
            vae,
            adata=data,
            labels_key="cellTypeName",
            unlabeled_category="Unknown",
        )
        lvae.train(max_epochs=20, n_samples_per_label=100)
        data.obsm["X_scANVI"] = lvae.get_latent_representation(data)


def main():

    p = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(
        description="Set the parameters for batch correction.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--read_path",
        type=lambda i: p.joinpath("data", i),
        default=p.joinpath("data", "asap", "haltere"),
        help="Path to the folder of the preprocessed data of batches from one tissue",
    )
    parser.add_argument(
        "--save_path",
        type=lambda i: p.joinpath("data", i),
        default=p.joinpath("data", "asap", "batch_corrected"),
        help="Path to save the batch-corrected data",
    )
    parser.add_argument(
        "--tissue",
        type=str,
        # default="unknown_tissue",
        default="haltere",
        help="The name of the tissue. \nIf not provided, will be set to 'unknown_tissue'.",
    )
    parser.add_argument(
        "--keys",
        "--list",
        nargs="+",
        type=str,
        default=["batch", "experiment_id"],
        help="The list of columns to regress on.",
    )
    parser.add_argument(
        "--batch_correction_method",
        type=str,
        # default="regress_out",
        default="scanvi",
        help="The batch correction method to use. \nIf not provided, will be set to 'scanvi'.",
    )
    args = parser.parse_args()

    files = [x for x in args.read_path.iterdir() if x.is_file()]
    data_list = [ad.read_h5ad(x) for x in files]
    file_names = [x.name for x in files]
    data_shape = [d.shape for d in data_list]
    logging.info(
        f"Read in batch data: {{file.name: data.shape}}: \n{dict(zip(file_names, data_shape))}"
    )
    data = data_list[0]
    data = ad.concat(data_list, merge="same")
    logging.info(f"Concatenated data.shape: {data.shape}")
    logging.info(f"Batch correction method: {args.batch_correction_method}")

    if args.batch_correction_method == "regress_out":
        batch_corrector = RegressOut(args.keys)
    elif args.batch_correction_method == "scanvi":
        batch_corrector = Scanvi(args.keys)
    else:
        raise ValueError("Please specify a valid batch correction method.")

    batch_corrector.correct_batch_effect(data)
    Path(args.save_path.joinpath(f"{args.tissue}")).mkdir(parents=True, exist_ok=True)
    save_path = args.save_path.joinpath(
        f"{args.tissue}", f"{args.tissue}_batch_corrected.h5ad"
    )
    data.write(save_path)
    logging.info(f"Batch-corrected data saved to {save_path}")


if __name__ == "__main__":
    main()
