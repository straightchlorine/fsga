"""Base evaluator class for fitness evaluation.

Adapted from knapsack-problem/knapsack/evaluators/evaluator.py
Key change: fitness = ML model performance (not item value)
"""

from abc import ABC, abstractmethod

import numpy as np


class Evaluator(ABC):
    """Abstract base class for chromosome fitness evaluation.

    In feature selection, fitness is typically ML model performance
    (accuracy, F1, etc.) on a validation set using the selected features.

    Subclasses must implement evaluate() method.

    Example:
        >>> class MyEvaluator(Evaluator):
        ...     def evaluate(self, chromosome):
        ...         # Select features where chromosome[i] == 1
        ...         # Train model on selected features
        ...         # Return validation performance
        ...         return fitness_score
    """

    @abstractmethod
    def evaluate(self, chromosome: np.ndarray) -> float:
        """Evaluate fitness of a chromosome.

        Args:
            chromosome: Binary array where 1 = include feature, 0 = exclude

        Returns:
            float: Fitness score (higher is better)

        Note:
            Must handle edge case where all genes are 0 (no features selected).
            Typically return 0.0 or small penalty in this case.
        """
        raise NotImplementedError("Method 'evaluate' must be implemented in a subclass.")

    def __str__(self):
        return f"{self.__class__.__name__}"
