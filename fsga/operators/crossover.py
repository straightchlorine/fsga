"""Base crossover operator class.

Ported from knapsack-problem/knapsack/operators/crossover.py
"""

from abc import ABC, abstractmethod

import numpy as np


class Crossover(ABC):
    """Abstract base class for crossover operators.

    Crossover (recombination) combines two parent chromosomes to create offspring.

    Example:
        >>> class MyCrossover(Crossover):
        ...     def crossover(self, parent1, parent2):
        ...         # Recombination logic
        ...         child1 = ...
        ...         child2 = ...
        ...         return np.array([child1, child2])
    """

    def __init__(self, dev: bool = False):
        """Initialize crossover operator.

        Args:
            dev: Debug mode (print crossover details)
        """
        self.dev = dev

    @abstractmethod
    def crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> np.ndarray:
        """Perform crossover to generate offspring.

        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome

        Returns:
            np.ndarray: Two children (shape: 2 × num_features)

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            "Method 'crossover' must be implemented in a subclass."
        )

    def __str__(self):
        return f"{self.__class__.__name__}"
