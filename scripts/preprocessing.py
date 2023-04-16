import argparse
from pathlib import Path
import logging
import scanpy as sc
import anndata as ad
import numpy as np
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)
sc.settings.verbosity = 2
sc.logging.print_header()


def main():
    p = Path(__file__).parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--read_path",
        type=lambda i: p.joinpath(i),
        default=p.joinpath("data-raw/ASAP41_final.h5ad"),
        help="Path to read the raw data",
    )
    parser.add_argument(
        "--save_path",
        type=lambda i: p.joinpath(i),
        default=p.joinpath("data"),
        help="Path to save the preprocessed data",
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
        "--mt_start",
        type=str,
        default="mt:",
        help="Mitochondrial genes names start with (mt_start)",
    )
    parser.add_argument(
        "--n_genes_qt",
        type=float,
        nargs=2,
        default=[0.02, 0.98],
        help="Set the lower and upper quantile limit of n_genes_by_counts",
    )
    parser.add_argument(
        "--pct_counts_mt_upper",
        type=float,
        default=20,
        help="Set the upper limit of pct_counts_mt",
    )
    parser.add_argument(
        "--normalize_target_sum",
        type=int,
        default=2000,
        help="Normalize the sum of counts in every cell to (normalize_target_sum) UMI",
    )
    parser.add_argument(
        "--tissue_col", type=str, default="asap_tissue", help="Column name of tissues"
    )
    parser.add_argument(
        "--batch_col", type=str, default="batch", help="Column name of batches"
    )
    args = parser.parse_args()

    data = ad.read_h5ad(args.read_path)
    logging.info(f"data.shape: {data.shape}")

    def split_data_by_category(data, category):
        category_counts = data.obs[category].value_counts()
        category_data_dict = {
            c: data[data.obs[category] == c] for c in category_counts.index
        }
        return category_counts, category_data_dict

    def preprocess_per_batch(batch):
        sc.pp.filter_cells(batch, min_genes=args.min_genes)
        # sc.pp.filter_genes(batch, min_cells=args.min_cells)
        batch.var["mt"] = batch.var.gene_name.str.startswith(args.mt_start)
        sc.pp.calculate_qc_metrics(
            batch, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        )
        n_genes = batch.obs.n_genes_by_counts
        lower_lim = np.quantile(n_genes, args.n_genes_qt[0])
        upper_lim = np.quantile(n_genes, args.n_genes_qt[1])
        batch = batch[
            (batch.obs.n_genes_by_counts.between(lower_lim, upper_lim))
            & (batch.obs.pct_counts_mt < args.pct_counts_mt_upper)
        ]
        sc.pp.normalize_total(batch, target_sum=args.normalize_target_sum)
        sc.pp.log1p(batch)
        sc.pp.highly_variable_genes(batch, min_mean=0.0125, max_mean=3, min_disp=0.5)
        batch.raw = batch
        return batch

    tissue_counts, tissue_dict = split_data_by_category(data, category=args.tissue_col)
    logging.info(f"tissue_counts: {tissue_counts}")
    tissue_pp_dict = {}
    for tissue_, tissue_data in tissue_dict.items():
        sc.pp.filter_genes(tissue_data, min_cells=args.min_cells)
        logging.info(
            f"tissue {tissue_} sc.pp.filter_genes -> tissue_data.shape = {tissue_data.shape}"
        )
        batch_counts, batch_data_dict = split_data_by_category(
            tissue_data, category=args.batch_col
        )
        logging.info(f"tissue {tissue_} batch_counts: {batch_counts}")
        Path(args.save_path.joinpath(f"{tissue_}")).mkdir(parents=True, exist_ok=True)
        batch_pp_dict = {}
        for batch_, batch_data in batch_data_dict.items():
            batch_pp = preprocess_per_batch(batch_data)
            logging.info(f"batch {batch_} batch_pp.shape = {batch_pp.shape}")
            batch_pp_dict[batch_] = batch_pp
            save_path = args.save_path.joinpath(
                f"{tissue_}", f"asap_{tissue_}_batch{batch_}_preprocessed.h5ad"
            )
            batch_pp.write(save_path)
            logging.info(f"batch{batch_}_preprocessed.h5ad saved to {save_path}")
        tissue_pp_dict[tissue_] = batch_pp_dict

    return tissue_pp_dict


if __name__ == "__main__":
    main()
