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


def regress_out(train, test, batch_key, covariate_keys):
    keys = covariate_keys.copy()
    keys.append(batch_key)
    keys_num = []
    keys_to_drop = []
    for key in keys:
        if not is_numeric_dtype(train.obs[key]):
            train.obs[f"{key}_num"] = pd.factorize(train.obs[key])[0]
            test.obs[f"{key}_num"] = pd.factorize(test.obs[key])[0]
            keys_num.append(f"{key}_num")
            keys_to_drop.append(f"{key}_num")
        else:
            keys_num.append(key)
    
    sc.pp.regress_out(train, keys_num, n_jobs=24)
    sc.tl.pca(train, n_comps=30)    
    train_regress_out = ad.AnnData(X=train.obsm["X_pca"])
    train_regress_out.obs = train.obs[["id", "batch_id", "y"]]
    
    sc.pp.regress_out(test, keys_num, n_jobs=24)
    sc.tl.pca(test, n_comps=30)
    test_regress_out = ad.AnnData(X=test.obsm["X_pca"])
    test_regress_out.obs = test.obs[["id", "batch_id", "y"]]

    return train_regress_out, test_regress_out


def scanvi_embedding(train, test, batch_key, covariate_keys):
    categorical_keys = []
    continuous_keys = []
    for key in covariate_keys:
        if not is_numeric_dtype(train.obs[key]):
            categorical_keys.append(key)
        else:
            continuous_keys.append(key)
    scvi.model.SCVI.setup_anndata(
        train,
        layer="counts",
        batch_key=batch_key,
        categorical_covariate_keys=categorical_keys,
        continuous_covariate_keys=continuous_keys,
    )
    vae_ref = scvi.model.SCVI(train, n_latent=30, n_layers=2)
    vae_ref.train(100)
    train.obs["labels_scanvi"] = train.obs["y"].values
    vae_ref_scan = scvi.model.SCANVI.from_scvi_model(
        vae_ref,
        adata=train,
        labels_key="labels_scanvi",
        unlabeled_category="Unknown",
    )
    # vae_ref_scan.view_anndata_setup(train)
    vae_ref_scan.train(max_epochs=25)
    # model_path = f"{DATADIR}/lvae_models/"
    # vae_ref_scan.save(model_path, overwrite=True)
    # train.obsm["X_scANVI"] = vae_ref_scan.get_latent_representation(train)
    # scvi.model.SCANVI.prepare_query_anndata(test, model_path)
    scvi.model.SCANVI.prepare_query_anndata(test, vae_ref_scan)
    vae_query = scvi.model.SCANVI.load_query_data(
        test,
        vae_ref_scan,
    )
    vae_query.train(
        max_epochs=100,
        plan_kwargs=dict(weight_decay=0.0),
        check_val_every_n_epoch=10,
    )
    train.obsm["X_scANVI"] = vae_query.get_latent_representation(train)
    test.obsm["X_scANVI"] = vae_query.get_latent_representation(test)
    train_scanvi = ad.AnnData(X=train.obsm["X_scANVI"])
    train.obs.loc[:, "scanvi_predict"] = vae_query.predict(train) 
    train_scanvi.obs = train.obs[["id", "batch_id", "y", "scanvi_predict"]]
    test_scanvi = ad.AnnData(X=test.obsm["X_scANVI"])
    test.obs.loc[:, "scanvi_predict"] = vae_query.predict(test)
    test_scanvi.obs = test.obs[["id", "batch_id", "y", "scanvi_predict"]]
    return train_scanvi, test_scanvi


def main():

    p = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(
        description="Set the parameters for batch correction.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--read_path",
        type=lambda i: p.joinpath("data", i),
        default=p.joinpath("data", "preprocessed", "wing_pp.h5ad"),
        help="Path to the folder of the preprocessed data.",
    )
    parser.add_argument(
        "--save_path",
        type=lambda i: p.joinpath("data", i),
        default=p.joinpath("data", "regress_out"),
        help="Path to save the batch-corrected data",
    )
    parser.add_argument(
        "--tissue",
        type=str,
        # default="unknown_tissue",
        default="wing",
        help="The name of the tissue. \nIf not provided, will be set to 'unknown_tissue'.",
    )
    parser.add_argument(
        "--batch_key",
        type=str,
        default="batch_id",
        help="The column name of the batch covariate.",
    )
    parser.add_argument(
        "--covariate_keys",
        "--list",
        nargs="+",
        type=str,
        default=["pct_counts_mt", "cell_cycle_diff"],
        help="The list of column names of the covariates.",
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
    logging.info(f"Read in data from {args.read_path}, data.shape = {data.shape}")
    logging.info(f"Batch correction method: {args.batch_correction_method}")
    
    
    batch_key = args.batch_key
    covariate_keys = args.covariate_keys
    batches = data.obs[args.batch_key].unique().to_list()
    train_dict, test_dict = {}, {}
    Path(
        args.save_path.joinpath(f"{args.tissue}", f"{args.batch_correction_method}")
    ).mkdir(parents=True, exist_ok=True)


    for i in range(0, len(batches)):
        test_set = data[data.obs["batch_id"] == batches[i]].copy()
        train_set = data[
            data.obs["batch_id"].isin([b for b in batches if b != batches[i]])
        ].copy()
        logging.info(
            f"Leave out batch {i}, train_set.X.shape = {train_set.X.shape}, test_set.X.shape = {test_set.X.shape}"
        )
        if args.batch_correction_method == "regress_out":
            train_dict[i], test_dict[i] = regress_out(
                train_set, test_set, batch_key, covariate_keys
            )
        elif args.batch_correction_method == "scanvi":
            train_dict[i], test_dict[i] = scanvi_embedding(
                train_set, test_set, batch_key, covariate_keys
            )
        train_save_path = args.save_path.joinpath(
            f"{args.tissue}", f"{args.batch_correction_method}", f"train_{i}.h5ad"
        )
        test_save_path = args.save_path.joinpath(
            f"{args.tissue}", f"{args.batch_correction_method}", f"test_{i}.h5ad"
        )
        train_dict[i].write_h5ad(train_save_path)
        test_dict[i].write_h5ad(test_save_path)

    logging.info(f"Train/test sets saved to {args.save_path}")


if __name__ == "__main__":
    main()
