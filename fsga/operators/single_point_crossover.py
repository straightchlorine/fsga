"""Single-point crossover operator.

Adapted from knapsack-problem/knapsack/operators/fixed_point_crossover.py
Simplified to always use random point (no fixed point option).
"""

import numpy as np

from fsga.operators.crossover import Crossover


class SinglePointCrossover(Crossover):
    """Single-point crossover: split at random point and swap tails.

    Randomly selects a crossover point, then creates children by swapping
    the genes after that point between parents.

    Example:
        Parent 1: [1, 0, 0, 1, 1]
        Parent 2: [0, 1, 1, 0, 0]
        Point: 3

        Child 1:  [1, 0, 0 | 0, 0]  (parent1[:3] + parent2[3:])
        Child 2:  [0, 1, 1 | 1, 1]  (parent2[:3] + parent1[3:])

    Properties:
        - Preserves long schemas (building blocks)
        - Lower disruption than uniform crossover
        - Good for problems where gene order matters

    Usage:
        >>> crossover = SinglePointCrossover()
        >>> parent1 = np.array([1, 0, 1, 0, 1])
        >>> parent2 = np.array([0, 1, 1, 1, 0])
        >>> children = crossover.crossover(parent1, parent2)
        >>> print(children.shape)  # (2, 5)
    """

    def __init__(self, dev: bool = False):
        """Initialize single-point crossover.

        Args:
            dev: Debug mode (print crossover details)
        """
        super().__init__(dev)

    def crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """Perform single-point crossover.

        Args:
            parent_a: First parent chromosome
            parent_b: Second parent chromosome

        Returns:
            np.ndarray: Two children (shape: 2 × num_features)

        Raises:
            ValueError: If parents have different sizes
        """
        if parent_a.size != parent_b.size:
            raise ValueError("Parents must have the same size.")

        # Pick random crossover point (between 1 and size-1)
        point = np.random.randint(1, parent_a.size)

        # Create children by swapping tails
        child_1 = np.concatenate([parent_a[:point], parent_b[point:]])
        child_2 = np.concatenate([parent_b[:point], parent_a[point:]])

        genes = [child_1, child_2]

        # Debug output
        if self.dev:
            print(f"SinglePointCrossover: Point={point}")
            print(f"Parent A: {parent_a}")
            print(f"Parent B: {parent_b}")
            print(f"Child 1:  {child_1}")
            print(f"Child 2:  {child_2}")
            print("=" * 40)

        return np.array(genes)
