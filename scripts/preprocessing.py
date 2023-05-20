import argparse
from pathlib import Path
import logging
import scanpy as sc
import anndata as ad

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
sc.settings.verbosity = 2
sc.logging.print_header()


def split_data_by_category(data, category):
    category_counts = data.obs[category].value_counts()
    category_data_dict = {
        c: data[data.obs[category] == c] for c in category_counts.index
    }
    return category_counts, category_data_dict


def preprocess_batch(batch, mt_gene_id, args):
    sc.pp.filter_cells(batch, min_genes=args.min_genes)
    sc.pp.filter_genes(batch, min_cells=args.min_cells)
    batch.var["mt"] = batch.var[args.gene_id_column].isin(mt_gene_id["ensembl_gene_id"])
    sc.pp.calculate_qc_metrics(
        batch, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    batch = batch[
        (batch.obs.n_genes_by_counts < args.n_genes_by_counts_upper)
        & (batch.obs.pct_counts_mt < args.pct_counts_mt_upper)
    ]
    batch.layers["counts"] = batch.X.copy()
    sc.pp.normalize_total(batch, target_sum=args.normalize_target_sum)
    sc.pp.log1p(batch)
    batch.raw = batch
    if args.filter_hvg:
        sc.pp.highly_variable_genes(
            batch,
            layer="counts",
            min_mean=args.filter_hvg_params[0],
            max_mean=args.filter_hvg_params[1],
            min_disp=args.filter_hvg_params[2],
        )
        batch = batch[:, batch.var.highly_variable].copy()

    return batch


def main():
    p = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(
        description="Set the parameters for preprocessing.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--read_path",
        type=lambda i: p.joinpath("data-raw", i),
        # default=p.joinpath("data-raw", "ASAP41_final.h5ad"),  # ASAP
        # default=p.joinpath("data-raw", "asap_haltere.h5ad"),  # ASAP haltere
        default=p.joinpath("data-raw", "SRP200614.h5ad"),  # Bgee
        help="Path to read the data file",
    )
    parser.add_argument(
        "--save_path",
        type=lambda i: p.joinpath("data", i),
        # default=p.joinpath("data", "asap"),  # ASAP
        default=p.joinpath("data", "bgee"),  # Bgee
        help="Path to save the preprocessed data",
    )
    parser.add_argument(
        "--organism",
        type=str,
        default="dmelanogaster",
        help="Organism (string) to query the mitochondrial gene symbols in ensembl biomart",
    )
    parser.add_argument(
        "--gene_id_column",
        type=str,
        default="gene_id",
        help="Column of gene id in var",
    )
    parser.add_argument(
        "--doublet_column",
        type=str,
        default="scrublet_call",
        help="Column that indicates singlet/doublet in obs",
    )
    parser.add_argument(
        "--singlet_value",
        type=str,
        # default="0.0",  # ASAP
        default="Singlet",  # Bgee
        help="Value in doublet column that indicates singlet",
    )
    parser.add_argument(
        "--tissue_column",
        type=str,
        # default="asap_tissue",  # ASAP
        default=None,  # Bgee
        help="Column name of tissues. \nIf not provided, will be set to None.",
    )
    parser.add_argument(
        "--tissue_name",
        type=str,
        # default="unknown_tissue",  # ASAP
        default="gut",  # Bgee
        help="If there is no tissue column \nand all the data is from the same known tissue,\nplease specify the tissue name. \nIf not provided, will be set to 'unknown_tissue'.",
    )
    parser.add_argument(
        "--batch_column",
        type=str,
        # default="batch",  # ASAP
        default=None,  # Bgee
        help="Column name of batches. \nIf not provided, will be set to None.",
    )
    parser.add_argument(
        "--batch_name",
        type=str,
        default="unknown",
        help="If there is no batch column \nand all the data is from the same known batch,\nplease specify the batch name. \nIf not provided, will be set to 'unknown_batch'.",
    )
    parser.add_argument(
        "--min_genes",
        type=int,
        default=200,
        help="Keep cells with >= (min_genes) genes",
    )
    parser.add_argument(
        "--min_cells",
        type=int,
        default=3,
        help="Keep genes that are found in >= (min_cells) cells",
    )
    parser.add_argument(
        "--n_genes_by_counts_upper",
        type=int,
        default=2500,
        help="Set the upper limit of the number of genes expressed in the counts matrix (n_genes_by_counts)",
    )
    parser.add_argument(
        "--pct_counts_mt_upper",
        type=float,
        default=20,
        help="Set the upper limit of the percentage of counts in mitochondrial genes",
    )
    parser.add_argument(
        "--normalize_target_sum",
        type=int,
        default=2000,
        help="Normalize the sum of counts in every cell to (normalize_target_sum) UMI",
    )
    parser.add_argument(
        "--filter_hvg",
        type=lambda i: True if i.lower() == "y" else False,
        default=False,
        help="Filter highly variable genes [y/n]",
    )
    parser.add_argument(
        "--filter_hvg_params",
        type=float,
        nargs=3,
        default=[0.0125, 3, 0.5],
        help="Set the parameters min_mean, max_mean, min_disp for identifying highly variable genes. \nDefault: 0.0125, 3, 0.5",
    )
    args = parser.parse_args()

    data = ad.read_h5ad(args.read_path)
    logging.info(f"data.shape: {data.shape}")
    data = data[data.obs[args.doublet_column] == args.singlet_value]
    logging.info(f"Doublets removed: data.shape: {data.shape}")

    if args.tissue_column is None:
        tissue_dict = {args.tissue_name: data}
        logging.info(f"tissue_counts: {args.tissue_name}  {data.shape[0]}")
    else:
        tissue_counts, tissue_dict = split_data_by_category(
            data, category=args.tissue_column
        )
        logging.info(f"tissue_counts: {tissue_counts}")
    mt_gene_id = sc.queries.mitochondrial_genes(
        args.organism, chromosome="mitochondrion_genome", attrname="ensembl_gene_id"
    )
    for tissue_, tissue_data in tissue_dict.items():
        if args.batch_column is None:
            batch_dict = {args.batch_name: tissue_data}
            logging.info(
                f"tissue {tissue_} batch_counts: {args.batch_name}  {tissue_data.shape[0]}"
            )
        else:
            batch_counts, batch_dict = split_data_by_category(
                tissue_data, category=args.batch_column
            )
            logging.info(f"tissue {tissue_} batch_counts: {batch_counts}")
        Path(args.save_path.joinpath(f"{tissue_}")).mkdir(parents=True, exist_ok=True)
        batch_pp_dict = {}
        for batch_, batch_data in batch_dict.items():
            batch_pp = preprocess_batch(batch_data, mt_gene_id, args)
            logging.info(f"batch {batch_} batch_pp.shape = {batch_pp.shape}")
            batch_pp_dict[batch_] = batch_pp
            save_path = args.save_path.joinpath(
                f"{tissue_}", f"{tissue_}_batch_{batch_}_pp.h5ad"
            )
            batch_pp.write(save_path)
            logging.info(f"Preprocessed data saved to {save_path}")


if __name__ == "__main__":
    main()
