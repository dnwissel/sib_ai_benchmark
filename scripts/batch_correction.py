import scanpy as sc
import anndata as ad
import argparse
import logging
from pathlib import Path
from abc import ABC, abstractmethod

sc.settings.verbosity = 2
sc.logging.print_header()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)


class BatchCorrector(ABC):
    def __init__(self, batch_key):
        self.batch_key = batch_key

    @abstractmethod
    def correct_batch_effect(self, data):
        pass


class RegressOut(BatchCorrector):
    def correct_batch_effect(self, data):
        sc.pp.regress_out(data, self.batch_key)


class Combat(BatchCorrector):
    def correct_batch_effect(self, data):
        sc.pp.combat(data, self.batch_key)


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
        "--batch_key",
        type=str,
        default="batch",
        help="The name of the column to identify batches.",
    )
    parser.add_argument(
        "--batch_correction_method",
        type=str,
        default="regress_out",
        # default="combat",
        help="The batch correction method to use. \nIf not provided, will be set to 'regress_out'.",
    )
    args = parser.parse_args()

    files = [x for x in args.read_path.iterdir() if x.is_file()]
    data_list = [ad.read_h5ad(x) for x in files]
    file_names = [x.name for x in files]
    data_shape = [d.shape for d in data_list]
    logging.info(
        f"Read in batch data: {{file.name: data.shape}}: \n{dict(zip(file_names, data_shape))}"
    )
    data = ad.concat(data_list, merge="same")
    logging.info(f"Concatenated data.shape: {data.shape}")
    logging.info(f"Batch correction method: {args.batch_correction_method}")

    if args.batch_correction_method == "regress_out":
        batch_corrector = RegressOut(args.batch_key)
    elif args.batch_correction_method == "combat":
        batch_corrector = Combat(args.batch_key)
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
