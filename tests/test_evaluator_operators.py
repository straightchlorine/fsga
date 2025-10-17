"""Unit tests for evaluator operators (F1, Balanced Accuracy)."""

import numpy as np
import pytest

from fsga.evaluators.f1_evaluator import F1Evaluator
from fsga.evaluators.balanced_accuracy_evaluator import BalancedAccuracyEvaluator
from fsga.ml.models import ModelWrapper


class TestF1Evaluator:
    """Test suite for F1Evaluator."""

    def test_initialization(self, iris_data_split):
        """Test evaluator initialization."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)

        evaluator = F1Evaluator(X_train, y_train, X_test, y_test, model, average="weighted")

        assert evaluator.X_train.shape == X_train.shape
        assert evaluator.y_train.shape == y_train.shape
        assert evaluator.X_val.shape == X_test.shape
        assert evaluator.y_val.shape == y_test.shape
        assert evaluator.model is model
        assert evaluator.average == "weighted"

    def test_evaluate_all_features(self, iris_data_split):
        """Test evaluation with all features selected."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = F1Evaluator(X_train, y_train, X_test, y_test, model, average="weighted")

        chromosome = np.ones(X_train.shape[1], dtype=int)
        fitness = evaluator.evaluate(chromosome)

        # F1-score should be between 0 and 1
        assert 0.0 <= fitness <= 1.0
        # With all features, should get reasonable F1
        assert fitness > 0.5

    def test_evaluate_no_features_returns_zero(self, iris_data_split):
        """Test that selecting zero features returns 0 fitness."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = F1Evaluator(X_train, y_train, X_test, y_test, model)

        chromosome = np.zeros(X_train.shape[1], dtype=int)
        fitness = evaluator.evaluate(chromosome)

        assert fitness == 0.0

    def test_evaluate_single_feature(self, iris_data_split):
        """Test evaluation with single feature selected."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = F1Evaluator(X_train, y_train, X_test, y_test, model)

        # Select only first feature
        chromosome = np.zeros(X_train.shape[1], dtype=int)
        chromosome[0] = 1

        fitness = evaluator.evaluate(chromosome)

        assert 0.0 <= fitness <= 1.0

    def test_evaluate_different_averaging_strategies(self, iris_data_split):
        """Test F1 with different averaging strategies."""
        X_train, X_test, y_train, y_test = iris_data_split
        chromosome = np.array([1, 1, 1, 1])

        for average in ["weighted", "macro", "micro"]:
            model = ModelWrapper("rf", n_estimators=10, random_state=42)
            evaluator = F1Evaluator(X_train, y_train, X_test, y_test, model, average=average)
            fitness = evaluator.evaluate(chromosome)

            assert 0.0 <= fitness <= 1.0, f"Failed for average={average}"

    def test_evaluate_binary_classification(self, iris_data_split):
        """Test F1 evaluator with binary classification."""
        X_train, X_test, y_train, y_test = iris_data_split

        # Convert to binary (class 0 vs rest)
        y_train_binary = (y_train == 0).astype(int)
        y_test_binary = (y_test == 0).astype(int)

        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = F1Evaluator(
            X_train, y_train_binary, X_test, y_test_binary, model, average="binary"
        )

        chromosome = np.array([1, 1, 1, 1])
        fitness = evaluator.evaluate(chromosome)

        assert 0.0 <= fitness <= 1.0

    def test_evaluate_imbalanced_dataset(self):
        """Test F1 on imbalanced dataset."""
        # Create highly imbalanced dataset (90% class 0, 10% class 1)
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.array([0] * 90 + [1] * 10)

        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = F1Evaluator(X_train, y_train, X_test, y_test, model, average="weighted")

        chromosome = np.array([1, 1, 1, 1, 1])
        fitness = evaluator.evaluate(chromosome)

        # Should work without crashing
        assert 0.0 <= fitness <= 1.0

    def test_str_representation(self, iris_data_split):
        """Test string representation."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", random_state=42)
        evaluator = F1Evaluator(X_train, y_train, X_test, y_test, model, average="macro")

        assert "F1Evaluator" in str(evaluator)
        assert "macro" in str(evaluator)

    def test_evaluate_deterministic(self, iris_data_split):
        """Test that evaluation is deterministic with same random seed."""
        X_train, X_test, y_train, y_test = iris_data_split
        chromosome = np.array([1, 0, 1, 0])

        model1 = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator1 = F1Evaluator(X_train, y_train, X_test, y_test, model1)
        fitness1 = evaluator1.evaluate(chromosome)

        model2 = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator2 = F1Evaluator(X_train, y_train, X_test, y_test, model2)
        fitness2 = evaluator2.evaluate(chromosome)

        assert fitness1 == fitness2


class TestBalancedAccuracyEvaluator:
    """Test suite for BalancedAccuracyEvaluator."""

    def test_initialization(self, iris_data_split):
        """Test evaluator initialization."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)

        evaluator = BalancedAccuracyEvaluator(X_train, y_train, X_test, y_test, model, adjusted=True)

        assert evaluator.X_train.shape == X_train.shape
        assert evaluator.y_train.shape == y_train.shape
        assert evaluator.X_val.shape == X_test.shape
        assert evaluator.y_val.shape == y_test.shape
        assert evaluator.model is model
        assert evaluator.adjusted is True

    def test_evaluate_all_features(self, iris_data_split):
        """Test evaluation with all features selected."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = BalancedAccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        chromosome = np.ones(X_train.shape[1], dtype=int)
        fitness = evaluator.evaluate(chromosome)

        # Balanced accuracy should be between 0 and 1
        assert 0.0 <= fitness <= 1.0
        # With all features, should get reasonable accuracy
        assert fitness > 0.5

    def test_evaluate_no_features_returns_zero(self, iris_data_split):
        """Test that selecting zero features returns 0 fitness."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = BalancedAccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        chromosome = np.zeros(X_train.shape[1], dtype=int)
        fitness = evaluator.evaluate(chromosome)

        assert fitness == 0.0

    def test_evaluate_single_feature(self, iris_data_split):
        """Test evaluation with single feature selected."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = BalancedAccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        # Select only first feature
        chromosome = np.zeros(X_train.shape[1], dtype=int)
        chromosome[0] = 1

        fitness = evaluator.evaluate(chromosome)

        assert 0.0 <= fitness <= 1.0

    def test_evaluate_adjusted_vs_unadjusted(self, iris_data_split):
        """Test difference between adjusted and unadjusted balanced accuracy."""
        X_train, X_test, y_train, y_test = iris_data_split
        chromosome = np.array([1, 1, 1, 1])

        model1 = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator_unadjusted = BalancedAccuracyEvaluator(
            X_train, y_train, X_test, y_test, model1, adjusted=False
        )
        fitness_unadj = evaluator_unadjusted.evaluate(chromosome)

        model2 = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator_adjusted = BalancedAccuracyEvaluator(
            X_train, y_train, X_test, y_test, model2, adjusted=True
        )
        fitness_adj = evaluator_adjusted.evaluate(chromosome)

        # Both should be valid
        assert 0.0 <= fitness_unadj <= 1.0
        assert fitness_adj >= -1.0  # Adjusted can be negative

    def test_evaluate_binary_classification(self, iris_data_split):
        """Test balanced accuracy with binary classification."""
        X_train, X_test, y_train, y_test = iris_data_split

        # Convert to binary
        y_train_binary = (y_train == 0).astype(int)
        y_test_binary = (y_test == 0).astype(int)

        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = BalancedAccuracyEvaluator(
            X_train, y_train_binary, X_test, y_test_binary, model
        )

        chromosome = np.array([1, 1, 1, 1])
        fitness = evaluator.evaluate(chromosome)

        assert 0.0 <= fitness <= 1.0

    def test_evaluate_imbalanced_dataset(self):
        """Test balanced accuracy on imbalanced dataset."""
        # Create highly imbalanced dataset
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.array([0] * 90 + [1] * 10)

        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = BalancedAccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        chromosome = np.array([1, 1, 1, 1, 1])
        fitness = evaluator.evaluate(chromosome)

        # Balanced accuracy handles imbalance better than regular accuracy
        assert 0.0 <= fitness <= 1.0

    def test_str_representation(self, iris_data_split):
        """Test string representation."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", random_state=42)

        evaluator_adj = BalancedAccuracyEvaluator(X_train, y_train, X_test, y_test, model, adjusted=True)
        assert "BalancedAccuracyEvaluator" in str(evaluator_adj)
        assert "adjusted" in str(evaluator_adj)

        evaluator_unadj = BalancedAccuracyEvaluator(X_train, y_train, X_test, y_test, model, adjusted=False)
        assert "unadjusted" in str(evaluator_unadj)

    def test_evaluate_deterministic(self, iris_data_split):
        """Test that evaluation is deterministic with same random seed."""
        X_train, X_test, y_train, y_test = iris_data_split
        chromosome = np.array([1, 0, 1, 0])

        model1 = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator1 = BalancedAccuracyEvaluator(X_train, y_train, X_test, y_test, model1)
        fitness1 = evaluator1.evaluate(chromosome)

        model2 = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator2 = BalancedAccuracyEvaluator(X_train, y_train, X_test, y_test, model2)
        fitness2 = evaluator2.evaluate(chromosome)

        assert fitness1 == fitness2

    def test_balanced_accuracy_better_for_imbalance(self):
        """Test that balanced accuracy is more appropriate for imbalanced data."""
        # Create very imbalanced dataset
        np.random.seed(42)
        X = np.random.rand(100, 5)
        # 95% class 0, 5% class 1
        y = np.array([0] * 95 + [1] * 5)

        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = BalancedAccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        chromosome = np.ones(5, dtype=int)
        bal_acc = evaluator.evaluate(chromosome)

        # Should handle imbalance gracefully
        assert 0.0 <= bal_acc <= 1.0
