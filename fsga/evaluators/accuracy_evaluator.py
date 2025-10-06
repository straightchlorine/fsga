"""Accuracy-based fitness evaluator.

Evaluates feature subsets by training ML model and measuring accuracy.
"""

import numpy as np

from fsga.evaluators.evaluator import Evaluator


class AccuracyEvaluator(Evaluator):
    """Evaluates chromosome fitness using classification accuracy.

    Trains a classifier on selected features and returns validation accuracy
    as the fitness score.

    Example:
        >>> from fsga.ml.models import ModelWrapper
        >>> model = ModelWrapper('rf', n_estimators=50, random_state=42)
        >>> evaluator = AccuracyEvaluator(X_train, y_train, X_val, y_val, model)
        >>>
        >>> chromosome = np.array([1, 0, 1, 1, 0])  # Select features 0, 2, 3
        >>> fitness = evaluator.evaluate(chromosome)
        >>> print(f"Accuracy with selected features: {fitness:.4f}")
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model,
    ):
        """Initialize accuracy evaluator.

        Args:
            X_train: Training features (shape: n_samples × n_features)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model: sklearn-compatible classifier (e.g., ModelWrapper)

        Note:
            Model is re-trained for each chromosome evaluation.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = model

    def evaluate(self, chromosome: np.ndarray) -> float:
        """Evaluate fitness of chromosome.

        Args:
            chromosome: Binary array (1=include feature, 0=exclude)

        Returns:
            float: Classification accuracy on validation set (0.0 to 1.0)
                Returns 0.0 if no features selected.

        Example:
            >>> chromosome = np.array([1, 1, 0, 0])  # Use first 2 features
            >>> accuracy = evaluator.evaluate(chromosome)
        """
        # Get indices of selected features
        selected_features = np.where(chromosome == 1)[0]

        # Penalty if no features selected
        if len(selected_features) == 0:
            return 0.0

        # Select features from data
        X_train_subset = self.X_train[:, selected_features]
        X_val_subset = self.X_val[:, selected_features]

        # Train model on selected features
        try:
            self.model.fit(X_train_subset, self.y_train)
            accuracy = self.model.score(X_val_subset, self.y_val)
            return accuracy
        except Exception as e:
            # If training fails (e.g., too few samples), return 0
            print(f"Warning: Model training failed: {e}")
            return 0.0
