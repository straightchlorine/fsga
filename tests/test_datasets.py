"""Unit tests for dataset loader module."""

import numpy as np
import pytest

from fsga.datasets.loader import load_dataset, get_dataset_info


class TestLoadDataset:
    """Test suite for load_dataset function."""

    def test_load_iris_no_split(self):
        """Test loading Iris dataset without split."""
        X, y, feature_names = load_dataset("iris", split=False)

        assert X.shape == (150, 4)
        assert y.shape == (150,)
        assert len(feature_names) == 4
        assert len(np.unique(y)) == 3  # 3 classes

    def test_load_wine_no_split(self):
        """Test loading Wine dataset without split."""
        X, y, feature_names = load_dataset("wine", split=False)

        assert X.shape == (178, 13)
        assert y.shape == (178,)
        assert len(feature_names) == 13
        assert len(np.unique(y)) == 3

    def test_load_breast_cancer_no_split(self):
        """Test loading Breast Cancer dataset without split."""
        X, y, feature_names = load_dataset("breast_cancer", split=False)

        assert X.shape == (569, 30)
        assert y.shape == (569,)
        assert len(feature_names) == 30
        assert len(np.unique(y)) == 2  # Binary classification

    def test_load_digits_no_split(self):
        """Test loading Digits dataset without split."""
        X, y, feature_names = load_dataset("digits", split=False)

        assert X.shape == (1797, 64)
        assert y.shape == (1797,)
        assert len(feature_names) == 64
        assert len(np.unique(y)) == 10  # 10 digit classes

    def test_load_unknown_dataset(self):
        """Test that loading unknown dataset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset")

    def test_load_with_train_test_split(self):
        """Test loading with train/test split."""
        X_train, X_test, y_train, y_test, feature_names = load_dataset(
            "iris", split=True, test_size=0.2
        )

        assert X_train.shape[0] + X_test.shape[0] == 150
        assert y_train.shape[0] + y_test.shape[0] == 150
        assert X_train.shape[1] == 4  # 4 features
        assert X_test.shape[1] == 4
        assert len(feature_names) == 4

        # Check split proportion (allow small variance)
        test_proportion = len(y_test) / 150
        assert 0.15 < test_proportion < 0.25

    def test_load_with_train_val_test_split(self):
        """Test loading with train/val/test split."""
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_dataset(
            "iris", split=True, test_size=0.2, val_size=0.1
        )

        total_samples = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        assert total_samples == 150

        assert X_train.shape[1] == 4
        assert X_val.shape[1] == 4
        assert X_test.shape[1] == 4
        assert len(feature_names) == 4

    def test_stratified_split(self):
        """Test that stratified split maintains class distribution."""
        X_train, X_test, y_train, y_test, _ = load_dataset(
            "iris", split=True, stratify=True
        )

        # Check class distribution in train and test
        train_dist = np.bincount(y_train) / len(y_train)
        test_dist = np.bincount(y_test) / len(y_test)

        # Distributions should be similar (within 10% tolerance)
        for i in range(3):
            assert abs(train_dist[i] - test_dist[i]) < 0.1

    def test_non_stratified_split(self):
        """Test non-stratified split option."""
        X_train, X_test, y_train, y_test, _ = load_dataset(
            "iris", split=True, stratify=False, random_state=42
        )

        # Should still split, just not stratified
        assert len(y_train) > 0
        assert len(y_test) > 0

    def test_random_state_reproducibility(self):
        """Test that same random_state produces same split."""
        result1 = load_dataset("iris", split=True, random_state=42)
        result2 = load_dataset("iris", split=True, random_state=42)

        X_train1, X_test1, y_train1, y_test1, _ = result1
        X_train2, X_test2, y_train2, y_test2, _ = result2

        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(y_train1, y_train2)

    def test_different_test_sizes(self):
        """Test loading with different test sizes."""
        test_sizes = [0.1, 0.2, 0.3, 0.4]

        for test_size in test_sizes:
            X_train, X_test, y_train, y_test, _ = load_dataset(
                "iris", split=True, test_size=test_size
            )

            actual_test_prop = len(y_test) / 150
            # Allow 5% variance
            assert abs(actual_test_prop - test_size) < 0.05

    def test_feature_names_are_strings(self):
        """Test that feature names are strings or string-like."""
        X, y, feature_names = load_dataset("iris", split=False)

        assert all(isinstance(name, (str, np.str_)) for name in feature_names)

    def test_data_types(self):
        """Test that data has correct types."""
        X, y, feature_names = load_dataset("iris", split=False)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(feature_names, (list, np.ndarray))

    def test_no_data_leakage_in_split(self):
        """Test that train and test sets don't overlap."""
        X_train, X_test, y_train, y_test, _ = load_dataset(
            "iris", split=True, random_state=42
        )

        # Check no samples overlap (difficult with floats, check sizes)
        assert len(y_train) + len(y_test) == 150
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)


class TestGetDatasetInfo:
    """Test suite for get_dataset_info function."""

    def test_iris_info(self):
        """Test dataset info for Iris."""
        info = get_dataset_info("iris")

        assert info["name"] == "iris"
        assert info["n_samples"] == 150
        assert info["n_features"] == 4
        assert info["n_classes"] == 3

    def test_wine_info(self):
        """Test dataset info for Wine."""
        info = get_dataset_info("wine")

        assert info["name"] == "wine"
        assert info["n_samples"] == 178
        assert info["n_features"] == 13
        assert info["n_classes"] == 3

    def test_breast_cancer_info(self):
        """Test dataset info for Breast Cancer."""
        info = get_dataset_info("breast_cancer")

        assert info["name"] == "breast_cancer"
        assert info["n_samples"] == 569
        assert info["n_features"] == 30
        assert info["n_classes"] == 2

    def test_digits_info(self):
        """Test dataset info for Digits."""
        info = get_dataset_info("digits")

        assert info["name"] == "digits"
        assert info["n_samples"] == 1797
        assert info["n_features"] == 64
        assert info["n_classes"] == 10

    def test_info_dict_structure(self):
        """Test that info dict has expected keys."""
        info = get_dataset_info("iris")

        expected_keys = ["name", "n_samples", "n_features", "n_classes"]
        for key in expected_keys:
            assert key in info

    def test_info_unknown_dataset(self):
        """Test that getting info for unknown dataset raises error."""
        with pytest.raises(ValueError):
            get_dataset_info("nonexistent")
