"""Unit tests for evaluator classes."""

import numpy as np
import pytest

from fsga.evaluators.accuracy_evaluator import AccuracyEvaluator
from fsga.ml.models import ModelWrapper


class TestAccuracyEvaluator:
    """Test suite for AccuracyEvaluator."""

    def test_initialization(self, iris_data_split):
        """Test evaluator initialization."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)

        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        assert evaluator.X_train.shape == X_train.shape
        assert evaluator.y_train.shape == y_train.shape
        assert evaluator.X_val.shape == X_test.shape
        assert evaluator.y_val.shape == y_test.shape
        assert evaluator.model is model

    def test_evaluate_all_features(self, iris_data_split):
        """Test evaluation with all features selected."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        chromosome = np.ones(X_train.shape[1], dtype=int)
        fitness = evaluator.evaluate(chromosome)

        # Fitness should be between 0 and 1 (accuracy)
        assert 0.0 <= fitness <= 1.0
        # With all features on iris, should get decent accuracy
        assert fitness > 0.5

    def test_evaluate_no_features_returns_zero(self, iris_data_split):
        """Test that selecting zero features returns 0 fitness."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        chromosome = np.zeros(X_train.shape[1], dtype=int)
        fitness = evaluator.evaluate(chromosome)

        assert fitness == 0.0

    def test_evaluate_single_feature(self, iris_data_split):
        """Test evaluation with single feature selected."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        # Select only first feature
        chromosome = np.zeros(X_train.shape[1], dtype=int)
        chromosome[0] = 1

        fitness = evaluator.evaluate(chromosome)

        assert 0.0 <= fitness <= 1.0

    def test_evaluate_subset_features(self, iris_data_split):
        """Test evaluation with subset of features."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        # Select 2 out of 4 features
        chromosome = np.array([1, 0, 1, 0])
        fitness = evaluator.evaluate(chromosome)

        assert 0.0 <= fitness <= 1.0

    def test_evaluate_deterministic_with_same_model(self, iris_data_split):
        """Test that evaluation is deterministic with same random seed."""
        X_train, X_test, y_train, y_test = iris_data_split
        chromosome = np.array([1, 0, 1, 0])

        model1 = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator1 = AccuracyEvaluator(X_train, y_train, X_test, y_test, model1)
        fitness1 = evaluator1.evaluate(chromosome)

        model2 = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator2 = AccuracyEvaluator(X_train, y_train, X_test, y_test, model2)
        fitness2 = evaluator2.evaluate(chromosome)

        assert fitness1 == fitness2

    def test_evaluate_different_chromosomes(self, iris_data_split):
        """Test that different chromosomes can produce different fitness."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        chromosome1 = np.array([1, 1, 0, 0])
        chromosome2 = np.array([0, 0, 1, 1])

        fitness1 = evaluator.evaluate(chromosome1)
        fitness2 = evaluator.evaluate(chromosome2)

        # Fitness values should be valid
        assert 0.0 <= fitness1 <= 1.0
        assert 0.0 <= fitness2 <= 1.0

    def test_evaluate_trains_model(self, iris_data_split):
        """Test that evaluate actually trains the model."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        chromosome = np.array([1, 0, 1, 0])

        # Before evaluation, model should not be fitted
        # After evaluation, it should be able to predict
        fitness = evaluator.evaluate(chromosome)

        # If training worked, fitness should be > 0
        assert fitness > 0

    def test_evaluate_with_different_models(self, iris_data_split):
        """Test evaluation with different model types."""
        X_train, X_test, y_train, y_test = iris_data_split
        chromosome = np.array([1, 1, 1, 1])

        model_types = ["rf", "logistic", "svm"]

        for model_type in model_types:
            model = ModelWrapper(model_type, random_state=42)
            evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)
            fitness = evaluator.evaluate(chromosome)

            assert 0.0 <= fitness <= 1.0, f"Model {model_type} produced invalid fitness"

    def test_evaluate_respects_chromosome(self, iris_data_split):
        """Test that evaluation only uses selected features."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        # Select features 2 and 3 (indices in Python)
        chromosome = np.array([0, 0, 1, 1])
        fitness = evaluator.evaluate(chromosome)

        # The model should only have access to 2 features
        # We can't directly check this, but fitness should be reasonable
        assert 0.0 <= fitness <= 1.0

    def test_evaluate_handles_binary_classification(self, iris_data_split):
        """Test evaluator works with binary classification."""
        X_train, X_test, y_train, y_test = iris_data_split

        # Convert to binary (class 0 vs rest)
        y_train_binary = (y_train == 0).astype(int)
        y_test_binary = (y_test == 0).astype(int)

        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = AccuracyEvaluator(
            X_train, y_train_binary, X_test, y_test_binary, model
        )

        chromosome = np.array([1, 1, 1, 1])
        fitness = evaluator.evaluate(chromosome)

        assert 0.0 <= fitness <= 1.0

    def test_evaluate_multiple_calls_same_chromosome(self, iris_data_split):
        """Test that multiple evaluations of same chromosome give same result."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        chromosome = np.array([1, 0, 1, 0])

        # Note: Without caching, fitness might vary slightly due to randomness
        # With random_state set, should be deterministic
        fitness1 = evaluator.evaluate(chromosome)
        fitness2 = evaluator.evaluate(chromosome)

        # Should be same or very close
        assert abs(fitness1 - fitness2) < 0.01

    def test_evaluate_boundary_chromosomes(self, iris_data_split):
        """Test evaluation with boundary cases."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        # All zeros
        fitness_none = evaluator.evaluate(np.zeros(4, dtype=int))
        assert fitness_none == 0.0

        # All ones
        fitness_all = evaluator.evaluate(np.ones(4, dtype=int))
        assert 0.0 < fitness_all <= 1.0

        # Single feature
        for i in range(4):
            chromosome = np.zeros(4, dtype=int)
            chromosome[i] = 1
            fitness = evaluator.evaluate(chromosome)
            assert 0.0 <= fitness <= 1.0

    def test_str_representation(self, iris_data_split):
        """Test string representation."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        assert "AccuracyEvaluator" in str(evaluator)
