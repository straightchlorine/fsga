"""Roulette wheel (fitness-proportionate) selection strategy.

Ported from knapsack-problem/knapsack/selectors/roulette_selector.py
Modified to return tuple instead of list for consistency.
"""

import numpy as np

from fsga.evaluators.evaluator import Evaluator
from fsga.selectors.selector import Selector


class RouletteSelector(Selector):
    """Roulette wheel selection: probability proportional to fitness.

    Assigns selection probability to each individual proportional to its
    fitness. Individuals with higher fitness have higher chance of selection.

    Also known as:
        - Fitness-proportionate selection
        - Roulette wheel selection

    Properties:
        - Stochastic selection (randomness involved)
        - Selection pressure increases with fitness variance
        - Can suffer from premature convergence if one individual dominates
        - Handles negative fitness by shifting values

    Example:
        Population with fitness [0.5, 0.3, 0.2]
        Probabilities: [50%, 30%, 20%]
        Individual 1 has 50% chance of being selected

    Usage:
        >>> selector = RouletteSelector(evaluator)
        >>> selector.population = population
        >>> parent1, parent2 = selector.select()
    """

    def __init__(self, evaluator: Evaluator, number_of_parents: int = 2):
        """Initialize roulette selector.

        Args:
            evaluator: Fitness evaluator
            number_of_parents: Number of parents to select (default: 2)
        """
        super().__init__()
        self.evaluator = evaluator
        self.number_of_parents = number_of_parents

    def select(self) -> tuple[np.ndarray, np.ndarray]:
        """Select parents using roulette wheel selection.

        Returns:
            tuple: Two parent chromosomes

        Note:
            If all fitness scores are non-positive, shifts them to be positive.
        """
        # Evaluate all chromosomes
        fitness_scores = np.array([self.evaluator.evaluate(c) for c in self.population])

        # Handle non-positive fitness (shift to make all positive)
        if (fitness_scores <= 0).all():
            fitness_scores = fitness_scores - fitness_scores.min() + 1e-10

        # Calculate selection probabilities (proportional to fitness)
        probabilities = fitness_scores / fitness_scores.sum()

        # Select parents based on probabilities
        selected_indices = np.random.choice(
            len(self.population), size=self.number_of_parents, p=probabilities
        )

        parents = [self.population[i] for i in selected_indices]
        return tuple(parents)

    def __str__(self):
        return "RouletteSelector"
