"""Model wrapper for sklearn classifiers.

Provides unified interface for different ML models.
"""


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


class ModelWrapper:
    """Unified interface for scikit-learn classifiers.

    Simplifies model instantiation and provides consistent API.

    Supported models:
        - 'rf': RandomForestClassifier
        - 'svm': Support Vector Classifier
        - 'logistic': LogisticRegression
        - 'knn': K-Nearest Neighbors
        - 'mlp': Multi-layer Perceptron (neural network)

    Example:
        >>> model = ModelWrapper('rf', n_estimators=100, random_state=42)
        >>> model.fit(X_train, y_train)
        >>> accuracy = model.score(X_test, y_test)

        >>> # Or get the underlying sklearn model
        >>> sklearn_model = model.model
    """

    SUPPORTED_MODELS = {
        "rf": RandomForestClassifier,
        "svm": SVC,
        "logistic": LogisticRegression,
        "knn": KNeighborsClassifier,
        "mlp": MLPClassifier,
    }

    def __init__(self, model_type: str = "rf", **kwargs):
        """Initialize model wrapper.

        Args:
            model_type: Type of model ('rf', 'svm', 'logistic', 'knn', 'mlp')
            **kwargs: Parameters passed to the sklearn model constructor

        Raises:
            ValueError: If model_type not supported

        Example:
            >>> model = ModelWrapper('rf', n_estimators=50, max_depth=10)
            >>> model = ModelWrapper('svm', kernel='rbf', C=1.0)
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Available: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_type = model_type
        self.model = self.SUPPORTED_MODELS[model_type](**kwargs)

    def fit(self, X, y):
        """Train the model.

        Args:
            X: Training features
            y: Training labels

        Returns:
            self: For method chaining
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions.

        Args:
            X: Features to predict

        Returns:
            Predicted labels
        """
        return self.model.predict(X)

    def score(self, X, y):
        """Calculate accuracy score.

        Args:
            X: Features
            y: True labels

        Returns:
            float: Accuracy (0.0 to 1.0)
        """
        return self.model.score(X, y)

    def __repr__(self):
        return f"ModelWrapper(type='{self.model_type}', model={self.model})"
