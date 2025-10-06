"""Tournament selection strategy.

Ported directly from knapsack-problem/knapsack/selectors/tournament_selector.py
No changes needed - works identically for feature selection.
"""

import numpy as np

from fsga.evaluators.evaluator import Evaluator
from fsga.selectors.selector import Selector


class TournamentSelector(Selector):
    """Tournament selection: best individual from random subset wins.

    Randomly selects k individuals, evaluates their fitness, and chooses
    the best one. Repeats to select multiple parents.

    Properties:
        - Adjustable selection pressure (via tournament_size)
        - Efficient (only evaluates k individuals, not whole population)
        - Works well with any fitness landscape

    Selection pressure:
        - tournament_size=2: Low pressure (more exploration)
        - tournament_size=5: Medium pressure (balanced)
        - tournament_size=10: High pressure (more exploitation)

    Example:
        >>> selector = TournamentSelector(evaluator, tournament_size=3)
        >>> selector.population = population  # Set by GeneticAlgorithm
        >>> parent1, parent2 = selector.select()
    """

    def __init__(
        self,
        evaluator: Evaluator,
        tournament_size: int = 3,
        number_of_parents: int = 2,
    ):
        """Initialize tournament selector.

        Args:
            evaluator: Fitness evaluator
            tournament_size: Number of individuals per tournament (default: 3)
            number_of_parents: Number of parents to select (default: 2)

        Note:
            tournament_size must be ≤ population_size
        """
        super().__init__()
        self.evaluator = evaluator
        self.tournament_size = tournament_size
        self.number_of_parents = number_of_parents

    def select(self) -> tuple[np.ndarray, np.ndarray]:
        """Select parents using tournament selection.

        Returns:
            tuple: Two parent chromosomes

        Raises:
            ValueError: If tournament_size > population_size
        """
        parents = []

        for _ in range(self.number_of_parents):
            # Validate tournament size
            if self.tournament_size > len(self.population):
                raise ValueError(
                    f"Tournament size ({self.tournament_size}) cannot be larger "
                    f"than population size ({len(self.population)})."
                )

            # Select random individuals for tournament
            tournament_indices = np.random.choice(
                range(len(self.population)),
                size=self.tournament_size,
                replace=False,
            )
            tournament = self.population[tournament_indices]

            # Find best individual in tournament
            winner = max(tournament, key=lambda chrom: self.evaluator.evaluate(chrom))
            parents.append(winner)

        return tuple(parents)
