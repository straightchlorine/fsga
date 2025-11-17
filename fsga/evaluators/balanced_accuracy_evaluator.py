"""Balanced accuracy based fitness evaluator.

Balanced accuracy adjusts for class imbalance by averaging recall
across classes, preventing bias toward majority class.
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from fsga.evaluators.evaluator import Evaluator


class BalancedAccuracyEvaluator(Evaluator):
    """Evaluates chromosome fitness using balanced accuracy.

    Balanced accuracy is the macro-average of recall scores per class.
    It's particularly useful for imbalanced datasets where standard
    accuracy would be misleading.

    Balanced Accuracy = (Sensitivity + Specificity) / 2  (for binary)
    Balanced Accuracy = mean(recall_per_class)  (for multi-class)

    Properties:
        - Robust to class imbalance
        - Range: 0.0 to 1.0 (higher is better)
        - Treats all classes equally
        - Random guessing gives ~0.5 for balanced datasets

    Example:
        >>> from fsga.ml.models import ModelWrapper
        >>> model = ModelWrapper('rf', n_estimators=50, random_state=42)
        >>> evaluator = BalancedAccuracyEvaluator(X_train, y_train, X_val, y_val, model)
        >>>
        >>> chromosome = np.array([1, 0, 1, 1, 0])
        >>> fitness = evaluator.evaluate(chromosome)
        >>> print(f"Balanced Accuracy: {fitness:.4f}")
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model,
        adjusted: bool = False,
    ):
        """Initialize balanced accuracy evaluator.

        Args:
            X_train: Training features (shape: n_samples × n_features)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model: sklearn-compatible classifier
            adjusted: If True, adjust for chance (default: False)
                Adjusted BA = (BA - 1/n_classes) / (1 - 1/n_classes)
                Makes random guessing score 0.0 instead of 1/n_classes

        Note:
            Model is re-trained for each chromosome evaluation.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = model
        self.adjusted = adjusted

    def evaluate(self, chromosome: np.ndarray) -> float:
        """Evaluate fitness of chromosome using balanced accuracy.

        Args:
            chromosome: Binary array (1=include feature, 0=exclude)

        Returns:
            float: Balanced accuracy on validation set (0.0 to 1.0)
                Returns 0.0 if no features selected or training fails.

        Example:
            >>> chromosome = np.array([1, 1, 0, 0])  # Use first 2 features
            >>> bal_acc = evaluator.evaluate(chromosome)
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
            y_pred = self.model.predict(X_val_subset)

            # Calculate balanced accuracy
            bal_acc = balanced_accuracy_score(self.y_val, y_pred, adjusted=self.adjusted)
            return float(bal_acc)

        except Exception as e:
            # If training fails, return 0
            print(f"Warning: Model training failed: {e}")
            return 0.0

    def __str__(self):
        adj_str = "adjusted" if self.adjusted else "unadjusted"
        return f"BalancedAccuracyEvaluator({adj_str})"
