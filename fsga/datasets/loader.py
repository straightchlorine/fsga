"""Dataset loaders for ML benchmarks.

Provides easy access to sklearn datasets with train/test splitting.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split


def load_dataset(
    name: str,
    split: bool = False,
    test_size: float = 0.2,
    val_size: float | None = None,
    stratify: bool = True,
    random_state: int = 42,
) -> (
    tuple[np.ndarray, np.ndarray, list]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]
):
    """Load dataset from sklearn.

    Args:
        name: Dataset name ('iris', 'wine', 'breast_cancer', 'digits')
        split: Whether to split into train/test (default: False)
        test_size: Test set proportion if split=True (default: 0.2)
        val_size: Validation set proportion from training (default: None)
        stratify: Maintain class distribution in splits (default: True)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        If split=False:
            (X, y, feature_names)
        If split=True and val_size=None:
            (X_train, X_test, y_train, y_test, feature_names)
        If split=True and val_size is set:
            (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)

    Example:
        >>> # Load without split
        >>> X, y, names = load_dataset('iris')

        >>> # Load with train/test split
        >>> X_train, X_test, y_train, y_test, names = load_dataset(
        ...     'iris', split=True
        ... )

        >>> # Load with train/val/test split
        >>> X_train, X_val, X_test, y_train, y_val, y_test, names = load_dataset(
        ...     'iris', split=True, val_size=0.1
        ... )
    """
    # Dataset loaders
    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")

    # Load data
    data = loaders[name]()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    if not split:
        return X, y, feature_names

    # Train/test split
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_param, random_state=random_state
    )

    if val_size is None:
        return X_train, X_test, y_train, y_test, feature_names

    # Train/val/test split
    # Adjust val_size to be proportion of remaining training data
    val_size_adjusted = val_size / (1 - test_size)
    stratify_param = y_train if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size_adjusted,
        stratify=stratify_param,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def get_dataset_info(name: str) -> dict:
    """Get metadata about a dataset.

    Args:
        name: Dataset name

    Returns:
        dict: Metadata including n_samples, n_features, n_classes

    Example:
        >>> info = get_dataset_info('iris')
        >>> print(info['n_features'])  # 4
    """
    X, y, _ = load_dataset(name, split=False)

    return {
        "name": name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
    }
