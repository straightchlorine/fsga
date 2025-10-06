"""Base selector class for parent selection.

Ported from knapsack-problem/knapsack/selectors/selector.py
"""

from abc import ABC, abstractmethod

import numpy as np

from fsga.evaluators.evaluator import Evaluator


class Selector(ABC):
    """Abstract base class for parent selection strategies.

    Selectors choose which chromosomes reproduce to create the next generation.

    Attributes:
        evaluator: Fitness evaluator
        population: Current population of chromosomes
    """

    def __init__(self):
        self._population: np.ndarray = None
        self.evaluator: Evaluator = None

    @abstractmethod
    def select(self) -> tuple[np.ndarray, np.ndarray]:
        """Select two parents for reproduction.

        Returns:
            tuple: Two parent chromosomes

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            "Method 'select' must be implemented in a subclass."
        )

    @property
    def population(self) -> np.ndarray:
        if self._population is None:
            raise ValueError("Population is not set.")
        return self._population

    @population.setter
    def population(self, population: np.ndarray):
        if not isinstance(population, np.ndarray) or len(population) == 0:
            raise ValueError("Population must be a non-empty numpy array.")
        self._population = population

    def validate_population_size(self, required_size: int):
        """Validate that population size meets minimum requirement.

        Args:
            required_size: Minimum population size needed

        Raises:
            ValueError: If population too small
        """
        if len(self.population) < required_size:
            raise ValueError(
                f"Population size ({len(self.population)}) must be at least "
                f"{required_size}."
            )

    def random_sample(self, size: int, replace: bool = False) -> np.ndarray:
        """Select random sample of individuals from population.

        Args:
            size: Number of individuals to sample
            replace: Whether to sample with replacement

        Returns:
            np.ndarray: Sampled chromosomes
        """
        indices = np.random.choice(len(self.population), size, replace=replace)
        return self.population[indices]

    def __str__(self):
        return f"{self.__class__.__name__}"
