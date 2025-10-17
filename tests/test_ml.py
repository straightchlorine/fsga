"""Unit tests for ML module (ModelWrapper)."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from fsga.ml.models import ModelWrapper


class TestModelWrapper:
    """Test suite for ModelWrapper."""

    def test_initialization_rf(self):
        """Test initialization with RandomForest."""
        model = ModelWrapper("rf", n_estimators=10, random_state=42)

        assert model.model_type == "rf"
        assert isinstance(model.model, RandomForestClassifier)
        assert model.model.n_estimators == 10
        assert model.model.random_state == 42

    def test_initialization_all_models(self):
        """Test initialization of all supported models."""
        # Test models that support random_state
        models_with_seed = ["rf", "svm", "logistic", "mlp"]
        for model_type in models_with_seed:
            model = ModelWrapper(model_type, random_state=42)
            assert model.model_type == model_type
            assert model.model is not None

        # Test knn without random_state
        model = ModelWrapper("knn")
        assert model.model_type == "knn"
        assert model.model is not None

    def test_initialization_unsupported_model(self):
        """Test that unsupported model raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            ModelWrapper("invalid_model")

    def test_initialization_with_kwargs(self):
        """Test that kwargs are passed to sklearn model."""
        model = ModelWrapper("rf", n_estimators=100, max_depth=5, random_state=42)

        assert model.model.n_estimators == 100
        assert model.model.max_depth == 5

    def test_fit(self, iris_data):
        """Test model fitting."""
        X, y = iris_data
        model = ModelWrapper("rf", n_estimators=10, random_state=42)

        result = model.fit(X, y)

        # fit should return self
        assert result is model
        # Model should be fitted
        assert hasattr(model.model, "classes_")

    def test_predict(self, iris_data):
        """Test making predictions."""
        X, y = iris_data
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert len(np.unique(predictions)) > 0

    def test_score(self, iris_data):
        """Test calculating accuracy score."""
        X, y = iris_data
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        model.fit(X, y)

        score = model.score(X, y)

        assert 0.0 <= score <= 1.0
        # Should get high accuracy on training data
        assert score > 0.9

    def test_fit_predict_score_workflow(self, iris_data_split):
        """Test complete workflow: fit, predict, score."""
        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)

        # Fit
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape

        # Score
        accuracy = model.score(X_test, y_test)
        assert 0.0 <= accuracy <= 1.0

    def test_different_model_types_work(self, iris_data_split):
        """Test that all model types can be fitted and scored."""
        X_train, X_test, y_train, y_test = iris_data_split

        # Models with random_state
        for model_type in ["rf", "logistic"]:
            model = ModelWrapper(model_type, random_state=42)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            assert 0.0 <= score <= 1.0, f"Model {model_type} produced invalid score"

        # KNN without random_state
        model = ModelWrapper("knn")
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        assert 0.0 <= score <= 1.0, "Model knn produced invalid score"

    def test_model_determinism_with_random_state(self, iris_data_split):
        """Test that models with same random_state produce same results."""
        X_train, X_test, y_train, y_test = iris_data_split

        model1 = ModelWrapper("rf", n_estimators=10, random_state=42)
        model1.fit(X_train, y_train)
        pred1 = model1.predict(X_test)

        model2 = ModelWrapper("rf", n_estimators=10, random_state=42)
        model2.fit(X_train, y_train)
        pred2 = model2.predict(X_test)

        assert np.array_equal(pred1, pred2)

    def test_repr(self):
        """Test string representation."""
        model = ModelWrapper("rf", n_estimators=10)

        repr_str = repr(model)
        assert "ModelWrapper" in repr_str
        assert "rf" in repr_str

    def test_supported_models_list(self):
        """Test that SUPPORTED_MODELS contains expected models."""
        expected = ["rf", "svm", "logistic", "knn", "mlp"]

        for model_type in expected:
            assert model_type in ModelWrapper.SUPPORTED_MODELS

    def test_model_attribute_access(self):
        """Test direct access to underlying sklearn model."""
        model = ModelWrapper("rf", n_estimators=10, random_state=42)

        # Should be able to access sklearn model directly
        sklearn_model = model.model
        assert isinstance(sklearn_model, RandomForestClassifier)
        assert sklearn_model.n_estimators == 10

    def test_fit_with_small_dataset(self):
        """Test fitting with very small dataset."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        model = ModelWrapper("rf", n_estimators=5, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == 3

    def test_binary_classification(self, iris_data):
        """Test model with binary classification."""
        X, y = iris_data
        # Convert to binary
        y_binary = (y == 0).astype(int)

        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        model.fit(X, y_binary)
        score = model.score(X, y_binary)

        assert 0.0 <= score <= 1.0

    def test_multiclass_classification(self, iris_data):
        """Test model with multiclass classification (3 classes)."""
        X, y = iris_data

        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        # Should have 3 classes
        assert len(np.unique(predictions)) <= 3

    def test_method_chaining(self, iris_data):
        """Test that fit returns self for method chaining."""
        X, y = iris_data

        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        predictions = model.fit(X, y).predict(X)

        assert len(predictions) == len(y)
