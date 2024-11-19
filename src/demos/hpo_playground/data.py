"""Data module."""

import os
import subprocess

from omegaconf import DictConfig
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.datasets import MNIST


def get_git_root() -> str:
    """Return the root directory of the current Git repository.

    Returns:
        The root directory of the current Git repository, or None if the command fails.
    """
    try:
        return subprocess.check_output(
            args=["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as e:
        print(f"Failed to get Git root: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return ""


def load_data() -> tuple[MNIST, MNIST]:
    """Load the MNIST dataset.

    It consists of a collection of handwritten digits from 0 to 9. Each digit is
    represented as a grayscale image of size 28x28 pixels. When the dataset is loaded,
    the images are transformed into tensors using the transforms.ToTensor() function.
    The resulting tensor has a shape of (1, 28, 28), where 1 represents the number of
    color channels (grayscale images have only one channel), and 28 represents the
    height and width of the image.
    The dataset also contains corresponding labels for each image, indicating the
    digit it represents. The labels are integers ranging from 0 to 9.
    Overall, the MNIST dataset consists of a collection of 60,000 training images
    and 10,000 test images, each with a shape of (1, 28, 28).

    Returns:
        train_set: The training set of the MNIST dataset.
        test_set: The test set of the MNIST dataset.
    """
    data_path = os.path.join(get_git_root(), "src", "demos", "hpo_playground", "data")
    train_set = MNIST(
        root=data_path, train=True, download=True, transform=transforms.ToTensor()
    )
    test_set = MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    return train_set, test_set


def create_data_loaders(
    cfg: DictConfig, train_set, test_set
) -> tuple[DataLoader, DataLoader]:
    """Create data loaders for the training and test sets.

    Args:
        cfg: The configuration object.
        train_set: The training set of the MNIST dataset.
        test_set: The test set of the MNIST dataset.

    Returns:
        train_loader: The data loader for the training set.
        test_loader: The data loader for the test set.
    """
    random_sampler = RandomSampler(
        data_source=train_set,
        replacement=True,
        num_samples=cfg.data.num_train_samples,
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.train.batch_size,
        sampler=random_sampler,
        num_workers=cfg.data.num_workers,
    )

    test_loader = DataLoader(dataset=test_set, num_workers=cfg.data.num_workers)

    return train_loader, test_loader
