"""Ranking selection strategy.

Ported from knapsack-problem/knapsack/selectors/ranking_selector.py
Modified to return tuple instead of list for consistency.
"""

import numpy as np

from fsga.evaluators.evaluator import Evaluator
from fsga.selectors.selector import Selector


class RankingSelector(Selector):
    """Ranking selection: probability based on fitness rank, not absolute value.

    Ranks individuals by fitness and assigns selection probability based on
    rank rather than raw fitness values. This reduces selection pressure
    compared to roulette selection.

    Properties:
        - Reduces premature convergence
        - More robust to fitness scaling issues
        - Selection pressure controlled by scale_factor
        - Works well when fitness values vary wildly

    Algorithm:
        1. Rank all individuals by fitness (best = rank 0)
        2. Calculate probabilities: p_i = exp(-rank_i / scale_factor)
        3. Normalize probabilities to sum to 1
        4. Select based on these probabilities

    Usage:
        >>> selector = RankingSelector(evaluator, scale_factor=1.5)
        >>> selector.population = population
        >>> parent1, parent2 = selector.select()
    """

    def __init__(
        self,
        evaluator: Evaluator,
        scale_factor: float = 1.5,
        number_of_parents: int = 2
    ):
        """Initialize ranking selector.

        Args:
            evaluator: Fitness evaluator
            scale_factor: Scaling factor for rank-based probabilities (default: 1.5)
                Higher values = more uniform selection
                Lower values = stronger selection pressure
            number_of_parents: Number of parents to select (default: 2)
        """
        super().__init__()
        self.evaluator = evaluator
        self.scale_factor = scale_factor
        self.number_of_parents = number_of_parents

    def select(self) -> tuple[np.ndarray, np.ndarray]:
        """Select parents using ranking selection.

        Returns:
            tuple: Two parent chromosomes

        Raises:
            ValueError: If population is not set or empty
        """
        # Evaluate all chromosomes
        fitness_scores = np.array([
            self.evaluator.evaluate(c) for c in self.population
        ])

        # Rank chromosomes (higher fitness = lower rank number)
        # argsort(-fitness) gives indices that would sort fitness descending
        # argsort of that gives ranks (0 = best)
        ranks = np.argsort(np.argsort(-fitness_scores))

        # Calculate probabilities based on ranks
        # Best individual (rank 0) has highest probability
        probabilities = np.exp(-ranks / self.scale_factor)
        probabilities /= probabilities.sum()

        # Select parents based on probabilities
        selected_indices = np.random.choice(
            len(self.population),
            size=self.number_of_parents,
            p=probabilities
        )

        parents = [self.population[i] for i in selected_indices]
        return tuple(parents)

    def __str__(self):
        return f"RankingSelector(scale={self.scale_factor})"
