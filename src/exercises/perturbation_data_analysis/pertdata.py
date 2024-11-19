"""Functionality for handling perturbation data."""

import os
import sys

import pandas as pd
import scanpy as sc
from anndata import AnnData

# Append the root of the Git repository to the path.
git_root = os.popen(cmd="git rev-parse --show-toplevel").read().strip()
sys.path.append(git_root)

from utils import download_file  # noqa: E402


class PertData:
    """Class for perturbation data.

    The perturbation data is stored in an `AnnData` object. Refer to
    https://anndata.readthedocs.io/en/latest/ for more information on `AnnData`.

    `AnnData` is specifically designed for matrix-like data. By this we mean that we
    have n observations, each of which can be represented as d-dimensional vectors,
    where each dimension corresponds to a variable or feature. Both the rows and columns
    of this matrix are special in the sense that they are indexed.

    For instance, in scRNA-seq data, each row corresponds to a cell with a barcode, and
    each column corresponds to a gene with a gene identifier.

    Attributes:
        name: The name of the dataset.
        path: The path where the dataset is stored.
        adata: The actual perturbation data.
    """

    def __init__(self) -> None:
        """Initialize the PertData object."""
        self.name: str = ""
        self.path: str = ""
        self.adata: AnnData = AnnData()

    def __str__(self) -> str:
        """Return a string representation of the PertData object."""
        adata_info = (
            "AnnData object with n_obs x n_vars = "
            f"{self.adata.shape[0]} x {self.adata.shape[1]}"
        )
        return (
            f"PertData object\n"
            f"    name: {self.name}\n"
            f"    path: {self.path}\n"
            f"    adata: {adata_info}"
        )

    @classmethod
    def from_repo(cls, name: str, save_dir: str) -> "PertData":
        """Load perturbation dataset from an online repository.

        Args:
            name: The name of the dataset to load (supported: "dixit", "adamson",
                "norman").
            save_dir: The directory to save the data.
        """
        instance = cls()
        instance.name = name
        instance.path = os.path.join(save_dir, instance.name)
        instance.adata = _load(dataset_name=name, dataset_dir=instance.path)
        instance.adata.obs["condition_fixed"] = generate_fixed_perturbation_labels(
            labels=instance.adata.obs["condition"]
        )
        return instance


def _load(dataset_name: str, dataset_dir: str) -> AnnData:
    """Load perturbation dataset.

    The following are the corresponding publications:
    - [Dixit et al., 2016](https://doi.org/10.1016/j.cell.2016.11.038)
    - [Adamson et al., 2016](https://doi.org/10.1016/j.cell.2016.11.048)
    - [Norman et al., 2019](https://doi.org/10.1126/science.aax4438)

    The datasets used here are those used in [GEARS](https://doi.org/10.1038/s41587-023-01905-6).
    These datasets all underwent the same preprocessing. First, each cell was normalized
    by total counts over all genes and then a log transformation was applied. To reduce
    the complexity, each dataset was restricted to only the 5000 most highly varying
    genes. Additionally, any perturbed gene is included that was not already accounted
    for in the set of most highly varying genes.

    The preprocessed datasets are available at the following URLs from the
    [GEARS code](https://github.com/snap-stanford/GEARS/blob/master/gears/pertdata.py):
    - Dixit et al., 2016: https://dataverse.harvard.edu/api/access/datafile/6154416
    - Adamson et al., 2016: https://dataverse.harvard.edu/api/access/datafile/6154417
    - Norman et al., 2019: https://dataverse.harvard.edu/api/access/datafile/6154020

    Copies of the preprocessed datasets are available at the following URLs from the
    ["Applied Machine Learning in Genomic Data Science" dataset on the Harvard Dataverse](https://doi.org/10.7910/DVN/ZSVS5X):
    - Dixit et al., 2016: https://dataverse.harvard.edu/api/access/datafile/10548302
    - Adamson et al., 2016: https://dataverse.harvard.edu/api/access/datafile/10548300
    - Norman et al., 2019: https://dataverse.harvard.edu/api/access/datafile/10548304

    Copies of the preprocessed datasets are available at the following URLs from the
    ["Applied Machine Learning in Genomic Data Science" dataset on the LUH Seafile server](https://seafile.cloud.uni-hannover.de/d/5d6029c6eaaf410c8b01/):
    - Dixit et al., 2016: https://seafile.cloud.uni-hannover.de/d/5d6029c6eaaf410c8b01/files/?p=%2Fperturbation_data_analysis%2Fdixit%2Fperturb_processed.h5ad&dl=1
    - Adamson et al., 2016: https://seafile.cloud.uni-hannover.de/d/5d6029c6eaaf410c8b01/files/?p=%2Fperturbation_data_analysis%2Fadamson%2Fperturb_processed.h5ad&dl=1
    - Norman et al., 2019: https://seafile.cloud.uni-hannover.de/d/5d6029c6eaaf410c8b01/files/?p=%2Fperturbation_data_analysis%2Fnorman%2Fperturb_processed.h5ad&dl=1

    Args:
        dataset_name: The name of the dataset to load (supported: "dixit", "adamson",
            "norman").
        dataset_dir: The directory to save the dataset.

    Returns:
        The perturbation data object.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    dataset_filename = os.path.join(dataset_dir, "perturb_processed.h5ad")

    if dataset_name == "dixit":
        url = "https://seafile.cloud.uni-hannover.de/d/5d6029c6eaaf410c8b01/files/?p=%2Fperturbation_data_analysis%2Fdixit%2Fperturb_processed.h5ad&dl=1"
    elif dataset_name == "adamson":
        url = "https://seafile.cloud.uni-hannover.de/d/5d6029c6eaaf410c8b01/files/?p=%2Fperturbation_data_analysis%2Fadamson%2Fperturb_processed.h5ad&dl=1"
    elif dataset_name == "norman":
        url = "https://seafile.cloud.uni-hannover.de/d/5d6029c6eaaf410c8b01/files/?p=%2Fperturbation_data_analysis%2Fnorman%2Fperturb_processed.h5ad&dl=1"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # If the dataset directory does not exist, create it and download the dataset.
    if not os.path.exists(path=dataset_dir):
        # Create dataset directory
        print(f"Creating dataset directory: {dataset_dir}")
        os.makedirs(name=dataset_dir)

        # Download the dataset.
        print(f"Downloading dataset: {dataset_name}")
        download_file(url=url, save_filename=dataset_filename)
    else:
        print(f"Dataset directory already exists: {dataset_dir}")

    # Load the dataset.
    print(f"Loading dataset: {dataset_name}")
    adata = sc.read_h5ad(filename=dataset_filename)

    return adata


def generate_fixed_perturbation_labels(labels: pd.Series) -> pd.Series:
    """Generate fixed perturbation labels.

    In the perturbation datasets, single-gene perturbations are expressed as:
    - ctrl+<gene1>
    - <gene1>+ctrl

    Double-gene perturbations are expressed as:
    - <gene1>+<gene2>

    However, in general, there could also be multi-gene perturbations, and they
    might be expressed as a string with additional superfluous "ctrl+" in the
    middle:
        - ctrl+<gene1>+ctrl+<gene2>+ctrl+<gene3>+ctrl

    Hence, we need to remove superfluous "ctrl+" and "+ctrl" matches, such that
    perturbations are expressed as:
    - <gene1> (single-gene perturbation)
    - <gene1>+<gene2> (double-gene perturbation)
    - <gene1>+<gene2>+...+<geneN> (multi-gene perturbation)

    Note: Control cells are not perturbed and are labeled as "ctrl". We do not
    modify these labels.

    Args:
        labels: The perturbation labels.

    Returns:
        The fixed perturbation labels.
    """
    # Remove "ctrl+" and "+ctrl" matches.
    labels_fixed = labels.str.replace(pat="ctrl+", repl="")
    labels_fixed = labels_fixed.str.replace(pat="+ctrl", repl="")

    return labels_fixed
