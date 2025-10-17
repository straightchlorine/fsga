"""Pytest configuration and shared fixtures for FSGA tests."""

import numpy as np
import pytest
from sklearn.datasets import load_iris


@pytest.fixture
def sample_chromosome():
    """Return a sample binary chromosome."""
    return np.array([1, 0, 1, 0])


@pytest.fixture
def sample_population():
    """Return a sample population of chromosomes."""
    return np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ])


@pytest.fixture
def iris_data():
    """Load and return Iris dataset."""
    X, y = load_iris(return_X_y=True)
    return X, y


@pytest.fixture
def iris_data_split():
    """Load Iris dataset with train/test split."""
    from sklearn.model_selection import train_test_split
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def simple_evaluator():
    """Return a simple mock evaluator that returns sum of chromosome."""
    class SimpleEvaluator:
        def evaluate(self, chromosome):
            # Simple fitness: number of features selected
            return float(chromosome.sum())

    return SimpleEvaluator()


@pytest.fixture
def dummy_model():
    """Return a simple dummy classifier."""
    from sklearn.dummy import DummyClassifier
    return DummyClassifier(strategy="most_frequent")
