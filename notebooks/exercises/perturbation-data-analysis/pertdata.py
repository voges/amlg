"""Functionality for handling perturbation data."""

import os
import sys

import pandas as pd
import scanpy as sc

# Append the root of the git repository to the path
git_root = os.popen(cmd="git rev-parse --show-toplevel").read().strip()
sys.path.append(git_root)

from utils import download_file, extract_zip


class PertData:
    """
    Class for perturbation data.

    The perturbation data is stored in an `AnnData` object.

    Refer to https://anndata.readthedocs.io/en/latest/ for more information on
    `AnnData`.

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
        self.name = None
        self.path = None
        self.adata = None

    def __str__(self) -> str:
        """Return a string representation of the PertData object."""
        return (
            f"PertData object\n"
            f"    name: {self.name}\n"
            f"    path: {self.path}\n"
            f"    adata: AnnData object with n_obs x n_vars = {self.adata.shape[0]} x {self.adata.shape[1]}"
        )

    @classmethod
    def from_geo(cls, name: str, save_dir: str) -> "PertData":
        """
        Load perturbation dataset from Gene Expression Omnibus (GEO).

        Args:
            name: The name of the dataset to load (supported: "dixit", "adamson",
                "norman").
            save_dir: The directory to save the data.
            fix_labels: Whether to compute and add fixed perturbation labels.
        """
        instance = cls()
        instance.name = name
        instance.path = os.path.join(save_dir, instance.name)
        instance.adata = _load(name=name, save_dir=save_dir)
        instance.adata.obs["condition_fixed"] = generate_fixed_perturbation_labels(
            labels=instance.adata.obs["condition"]
        )
        return instance


def _load(name: str, save_dir: str) -> sc.AnnData:
    """
    Load perturbation dataset.

    The following are the [Gene Expression Omnibus](https://www.ncbi.nlm.nih.gov/geo/)
    accession numbers used:
    - Dixit et al., 2016: [GSE90063](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063)
    - Adamson et al., 2016: [GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546)
    - Norman et al., 2019: [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344)

    The following are the DOIs of the corresponding publications:
    - Dixit et al., 2016: https://doi.org/10.1016/j.cell.2016.11.038
    - Adamson et al., 2016: https://doi.org/10.1016/j.cell.2016.11.048
    - Norman et al., 2019: https://doi.org/10.1126/science.aax4438

    Args:
        name: The name of the dataset to load (supported: "dixit", "adamson",
            "norman").
        save_dir: The directory to save the data.

    Returns:
        The perturbation data object.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    if name == "dixit":
        url = "https://dataverse.harvard.edu/api/access/datafile/6154416"
        filename = "dixit/perturb_processed.h5ad"
    elif name == "adamson":
        url = "https://dataverse.harvard.edu/api/access/datafile/6154417"
        filename = "adamson/perturb_processed.h5ad"
    elif name == "norman":
        url = "https://dataverse.harvard.edu/api/access/datafile/6154020"
        filename = "norman/perturb_processed.h5ad"
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Create the data directory if it does not exist
    if not os.path.exists(path=save_dir):
        print(f"Creating data directory: {save_dir}")
        os.makedirs(name=save_dir)
    else:
        print(f"Data directory already exists: {save_dir}")

    # Download the dataset
    print(f"Downloading dataset: {name}")
    zip_filename = os.path.join(save_dir, f"{name}.zip")
    download_file(url=url, save_filename=zip_filename)

    # Extract the dataset
    print(f"Extracting dataset: {name}")
    extract_zip(zip_path=zip_filename, extract_dir=os.path.join(save_dir, name))

    # Load the dataset
    print(f"Loading dataset: {name}")
    adata = sc.read_h5ad(filename=os.path.join(save_dir, name, filename))

    return adata


def generate_fixed_perturbation_labels(labels: pd.Series) -> pd.Series:
    """
    Generate fixed perturbation labels.

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
    # Remove "ctrl+" and "+ctrl" matches
    labels_fixed = labels.str.replace(pat="ctrl+", repl="")
    labels_fixed = labels_fixed.str.replace(pat="+ctrl", repl="")

    return labels_fixed
