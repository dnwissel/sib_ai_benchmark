import anndata as ad
import numpy as np
import pandas as pd
import scanpy
from scipy.sparse import csr_matrix, hstack, vstack

bgee = scanpy.read_h5ad("./data-raw/SRP200614.h5ad.txt")
asap = scanpy.read_h5ad("./data-raw/asap.h5ad")
asap_mask = np.logical_and(
    asap.obs["cellTypeId"] != "NA",
    asap.obs["scrublet_call"].values.astype(str) != "1.0",
)
asap = asap[asap_mask, :]

asap.obs["cellTypeId"] = asap.obs.cellTypeId.astype(str)

offenders = np.unique(
    asap.var["gene_id"][
        np.where(
            np.isin(
                asap.var["gene_id"].values,
                np.unique(asap.var.gene_id, return_counts=True)[0][
                    np.where(
                        np.unique(asap.var.gene_id, return_counts=True)[1] > 1
                    )[0]
                ],
            )
        )[0]
    ]
)
for offender in offenders:
    asap.var.gene_id[np.where(asap.var.gene_id == offender)[0][1]] = (
        asap.var.gene_id[np.where(asap.var.gene_id == offender)[0][1]]
        + "_duplicated"
    )
genes_not_in_bgee = np.setdiff1d(asap.var.gene_id, bgee.var.gene_id)
genes_not_in_asap = np.setdiff1d(bgee.var.gene_id, asap.var.gene_id)
asap.var.drop("gene_name", axis=1, inplace=True)


overall_genes = np.sort(
    np.unique(np.union1d(asap.var.gene_id, bgee.var.gene_id))
)
asap_gene_sorted_ix = np.argsort(
    np.concatenate([asap.var.gene_id, genes_not_in_asap])
)
bgee_gene_sorted_ix = np.argsort(
    np.concatenate([bgee.var.gene_id, genes_not_in_bgee])
)

new_data = vstack(
    (
        hstack(
            (
                asap.X,
                csr_matrix(
                    np.zeros((asap.X.shape[0], genes_not_in_asap.shape[0]))
                ),
            )
        )[:, asap_gene_sorted_ix],
        csr_matrix(
            hstack(
                (
                    bgee.X,
                    csr_matrix(
                        np.zeros((bgee.X.shape[0], genes_not_in_bgee.shape[0]))
                    ),
                )
            )
        )[:, bgee_gene_sorted_ix],
    )
)
new_genes = pd.DataFrame({"gene_id": overall_genes})
meta = pd.DataFrame(
    {
        "tissue": np.concatenate(
            [asap.obs.asap_tissue, np.repeat("gut", bgee.obs.shape[0])]
        ),
        "batch_id": np.concatenate(
            [
                asap.obs.batch,
                np.repeat(
                    (np.max(asap.obs.batch.astype(int)) + 5)
                    .astype(int)
                    .astype(str),
                    bgee.obs.shape[0],
                ),
            ]
        ),
        "y": np.concatenate([asap.obs.cellTypeId, bgee.obs.cellTypeId]),
    }
)

combined_adata = ad.AnnData(new_data)
combined_adata.obs = meta
combined_adata.var = new_genes

types_to_keep = np.unique(combined_adata.obs.y, return_counts=True)[0][
    np.where(np.unique(combined_adata.obs.y, return_counts=True)[1] > 20)[0]
]
types_to_keep = np.array([i for i in types_to_keep if "[" not in i])
combined_adata = combined_adata[
    np.isin(combined_adata.obs.y, types_to_keep), :
]
combined_adata.write_h5ad("./data-raw/data_unionized.h5ad")

for tissue in np.unique(combined_adata.obs.tissue):
    combined_adata[combined_adata.obs.tissue == tissue, :].write_h5ad(
        f"./data-raw/data_{tissue}_unionized.h5ad"
    )
