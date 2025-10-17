"""Elitism selection strategy.

Ported from knapsack-problem/knapsack/selectors/elitism_selector.py
Modified to return tuple instead of list for consistency.
"""

import numpy as np

from fsga.evaluators.evaluator import Evaluator
from fsga.selectors.selector import Selector


class ElitismSelector(Selector):
    """Elitism selection: always select the best individuals.

    Deterministically selects the top N individuals by fitness. Guarantees
    that the best solutions are preserved across generations.

    Properties:
        - Deterministic (no randomness)
        - Strong selection pressure
        - Risk of premature convergence
        - Ensures best solutions never lost
        - Low diversity

    Usage:
        >>> selector = ElitismSelector(evaluator, n_elite=2)
        >>> selector.population = population
        >>> parent1, parent2 = selector.select()
    """

    def __init__(self, evaluator: Evaluator, n_elite: int = 2):
        """Initialize elitism selector.

        Args:
            evaluator: Fitness evaluator
            n_elite: Number of elite individuals to select (default: 2)
        """
        super().__init__()
        self.evaluator = evaluator
        self.n_elite = n_elite

    def select(self) -> tuple[np.ndarray, np.ndarray]:
        """Select top N elite parents.

        Returns:
            tuple: Two best chromosomes (by fitness)

        Note:
            Always returns the same parents (the best ones) until
            population changes.
        """
        # Sort population by fitness (descending)
        sorted_population = sorted(
            self.population,
            key=lambda c: self.evaluator.evaluate(c),
            reverse=True
        )

        # Return top n_elite individuals
        elite = sorted_population[:self.n_elite]

        # Ensure we return exactly 2 (pad with best if needed)
        while len(elite) < 2:
            elite.append(elite[0])

        return tuple(elite[:2])

    def __str__(self):
        return f"ElitismSelector(n_elite={self.n_elite})"
