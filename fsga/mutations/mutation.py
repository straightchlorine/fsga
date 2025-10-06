"""Base mutation operator class.

Ported from knapsack-problem/knapsack/mutations/mutation.py
"""

from abc import ABC, abstractmethod

import numpy as np


class Mutation(ABC):
    """Abstract base class for mutation operators.

    Mutation introduces random variation into offspring chromosomes.

    Attributes:
        probability: Mutation probability per gene (0.0 to 1.0)
    """

    def __init__(self, probability: float):
        """Initialize mutation operator.

        Args:
            probability: Mutation rate (0.0 to 1.0)

        Raises:
            ValueError: If probability not in [0, 1]
        """
        self.probability = probability

    @abstractmethod
    def mutate(self, population: np.ndarray, generation: int = 0) -> np.ndarray:
        """Apply mutation to population.

        Args:
            population: Population of chromosomes (2D array)
            generation: Current generation (for dynamic mutation)

        Returns:
            np.ndarray: Mutated population

        Raises:
            NotImplementedError: Must be implemented by subclass

        Note:
            Must not modify input population (copy first).
        """
        raise NotImplementedError(
            "Method 'mutate' must be implemented in a subclass."
        )

    @property
    def probability(self) -> float:
        return self._probability

    @probability.setter
    def probability(self, probability: float):
        if not (0 <= probability <= 1):
            raise ValueError("Mutation probability must be between 0 and 1.")
        self._probability = probability

    def __str__(self):
        return f"{self.__class__.__name__}(p={self.probability})"
