"""F1-score based fitness evaluator.

F1-score is the harmonic mean of precision and recall,
useful for imbalanced datasets where accuracy can be misleading.
"""

import numpy as np
from sklearn.metrics import f1_score

from fsga.evaluators.evaluator import Evaluator


class F1Evaluator(Evaluator):
    """Evaluates chromosome fitness using F1-score.

    F1-score balances precision and recall, making it better than accuracy
    for imbalanced datasets. Particularly useful when both false positives
    and false negatives are important.

    F1 = 2 * (precision * recall) / (precision + recall)

    Properties:
        - Better for imbalanced datasets than accuracy
        - Range: 0.0 to 1.0 (higher is better)
        - Harmonic mean penalizes extreme values
        - Supports multi-class via averaging strategy

    Example:
        >>> from fsga.ml.models import ModelWrapper
        >>> model = ModelWrapper('rf', n_estimators=50, random_state=42)
        >>> evaluator = F1Evaluator(X_train, y_train, X_val, y_val, model, average='weighted')
        >>>
        >>> chromosome = np.array([1, 0, 1, 1, 0])
        >>> fitness = evaluator.evaluate(chromosome)
        >>> print(f"F1-score: {fitness:.4f}")
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model,
        average: str = "weighted"
    ):
        """Initialize F1 evaluator.

        Args:
            X_train: Training features (shape: n_samples × n_features)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model: sklearn-compatible classifier
            average: Averaging strategy for multi-class:
                - 'binary': Binary classification (default for 2 classes)
                - 'micro': Global precision/recall
                - 'macro': Unweighted mean across classes
                - 'weighted': Weighted by support (default)
                - 'samples': For multilabel classification

        Note:
            Model is re-trained for each chromosome evaluation.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = model
        self.average = average

    def evaluate(self, chromosome: np.ndarray) -> float:
        """Evaluate fitness of chromosome using F1-score.

        Args:
            chromosome: Binary array (1=include feature, 0=exclude)

        Returns:
            float: F1-score on validation set (0.0 to 1.0)
                Returns 0.0 if no features selected or training fails.

        Example:
            >>> chromosome = np.array([1, 1, 0, 0])  # Use first 2 features
            >>> f1 = evaluator.evaluate(chromosome)
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

            # Calculate F1-score
            f1 = f1_score(self.y_val, y_pred, average=self.average, zero_division=0.0)
            return float(f1)

        except Exception as e:
            # If training fails, return 0
            print(f"Warning: Model training failed: {e}")
            return 0.0

    def __str__(self):
        return f"F1Evaluator(average='{self.average}')"
