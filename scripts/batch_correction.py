import scvi
import scanpy as sc
import anndata as ad
import argparse
import logging
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pathlib import Path

sc.settings.verbosity = 2
sc.logging.print_header()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)


def regress_out(data, keys):
    keys_num = []
    keys_to_drop = []
    for key in keys:
        if not is_numeric_dtype(data.obs[key]):
            data.obs[f"{key}_num"] = pd.factorize(data.obs[key])[0]
            keys_num.append(f"{key}_num")
            keys_to_drop.append(f"{key}_num")
        else:
            keys_num.append(key)
    sc.pp.regress_out(data, keys_num)
    data.obs.drop(columns=keys_to_drop, inplace=True)
    new_anndata = ad.AnnData(X=data.X)
    new_anndata.obs = data.obs[["cellTypeName"]]
    new_anndata.var = data.var[["gene_id"]]

    return new_anndata


def scanvi_embed(data, keys):
    scvi.model.SCVI.setup_anndata(data, layer="counts", batch_key=keys[0])
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
    logging.info(
        f"Finished batch correction with scANVI, \n projected to {data.obsm['X_scANVI'].shape[1]} latent dimensions."
    )
    new_anndata = ad.AnnData(X=data.obsm["X_scANVI"])
    new_anndata.obs = data.obs[["cellTypeName"]]

    return new_anndata, lvae


def main():

    p = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(
        description="Set the parameters for batch correction.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--read_path",
        type=lambda i: p.joinpath("data", i),
        default=p.joinpath("data", "asap", "preprocessed", "antenna_pp.h5ad"),
        help="Path to the folder of the preprocessed data.",
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
        default="antenna",
        help="The name of the tissue. \nIf not provided, will be set to 'unknown_tissue'.",
    )
    parser.add_argument(
        "--keys",
        "--list",
        nargs="+",
        type=str,
        default=["batch", "pct_counts_mt", "cell_cycle_diff"],
        help="The list of columns to regress on.",
    )
    parser.add_argument(
        "--batch_correction_method",
        type=str,
        default="regress_out",
        # default="scanvi",
        help="The batch correction method to use.",
    )
    args = parser.parse_args()
    data = ad.read_h5ad(args.read_path)
    logging.info(f"Read in data {args.read_path}, data shape: {data.shape}")
    # Outer CV split train/test sets
    train_batch = data.obs["batch"].unique()[:-1]
    test_batch = data.obs["batch"].unique()[-1]
    train_set = data[data.obs["batch"].isin(train_batch)]
    test_set = data[data.obs["batch"] == test_batch]
    logging.info(f"Batch correction method: {args.batch_correction_method}")

    if args.batch_correction_method == "regress_out":
        train_bc = regress_out(train_set, args.keys)
        test_bc = regress_out(test_set, args.keys)
    elif args.batch_correction_method == "scanvi":
        train_bc, lvae_model = scanvi_embed(train_set, args.keys)
        test_em = lvae_model.get_latent_representation(test_set)
        test_bc = ad.AnnData(X=test_em.obsm["X_scANVI"])
        test_bc.obs = test_em.obs[["cellTypeName"]]
    else:
        raise ValueError("Please specify a valid batch correction method.")

    Path(args.save_path.joinpath(f"{args.tissue}", f"{args.batch_correction_method}")).mkdir(parents=True, exist_ok=True)
    train_save_path = args.save_path.joinpath(
        f"{args.tissue}", f"{args.tissue}_train.h5ad"
    )
    test_save_path = args.save_path.joinpath(
        f"{args.tissue}", f"{args.tissue}_test.h5ad"
    )
    train_bc.write(train_save_path)
    test_bc.write(test_save_path)
    logging.info(f"Train/test sets saved to {args.save_path}")


if __name__ == "__main__":
    main()
