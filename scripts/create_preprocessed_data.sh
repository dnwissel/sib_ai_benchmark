for tissue in  'wing' \
    # 'body' 'body_wall' 'fat_body' 'gut' 'haltere' \
    # 'head' 'heart' 'leg' 'male_reproductive_glands' \
    # 'malpighian_tubule' 'oenocyte' 'ovary' 'proboscis_and_maxpalp' \
    # 'testis' 'trachea' 'antenna' ; \
    do

    python ./scripts/preprocessing.py \
        # --read_path "/Volumes/Backup/work/sib_ai_benchmark/data-raw/data_${tissue}_unionized.h5ad" \
        --read_path "data-raw/data_${tissue}_unionized.h5ad" \
        # --save_path "/Volumes/Backup/work/sib_ai_benchmark/data" \
        --save_path "data" \
        --organism "dmelanogaster" \
        --gene_id_column "gene_id" \
        --doublet_column "scrublet_call" \
        --doublet "1.0" \
        --tissue_column "tissue" \
        --batch_column "batch_id" \
        --min_genes 200 \
        --min_cells 3 \
        --n_genes_by_counts_upper 100000 \
        --pct_counts_mt_upper 25 \
        --normalize_target_sum 10000 \
        # --cell_cycle_genes_reference "/Volumes/Backup/work/sib_ai_benchmark/data-raw/Drosophila_melanogaster.csv" \
        --cell_cycle_genes_reference "data-raw/Drosophila_melanogaster.csv" \
        --cell_cycle_genes_column "gene_id" \
        --filter_hvg "n"

done
