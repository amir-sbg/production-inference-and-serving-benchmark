"""Dataset loading for the benchmark workload."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    """A deterministic train/test split used by both training and benchmarking."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_dataset(test_size: float = 0.2, random_state: int = 42) -> DatasetSplit:
    """Return the scikit-learn digits dataset with a stratified split.

    The workload has 64 numeric features and 10 classes. It is intentionally
    small so a reviewer can reproduce the benchmark on a laptop or in CI.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=test_size,
        random_state=random_state,
        stratify=digits.target,
    )
    return DatasetSplit(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

