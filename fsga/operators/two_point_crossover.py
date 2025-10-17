"""Two-point crossover operator.

Simplified version of multi-point crossover with exactly 2 points.
"""

import numpy as np

from fsga.operators.crossover import Crossover


class TwoPointCrossover(Crossover):
    """Two-point crossover: split at two random points and swap middle segment.

    Randomly selects two crossover points, then creates children by swapping
    the genes between those points.

    Example:
        Parent 1: [1, 0, 0, 1, 1, 0]
        Parent 2: [0, 1, 1, 0, 0, 1]
        Points: 2, 4

        Child 1:  [1, 0 | 1, 0 | 1, 0]  (p1[:2] + p2[2:4] + p1[4:])
        Child 2:  [0, 1 | 0, 1 | 0, 1]  (p2[:2] + p1[2:4] + p2[4:])

    Properties:
        - Better than single-point for preserving schemas
        - Swaps middle segment between parents
        - Good balance between exploration and exploitation

    Usage:
        >>> crossover = TwoPointCrossover()
        >>> parent1 = np.array([1, 0, 1, 0, 1, 0])
        >>> parent2 = np.array([0, 1, 1, 1, 0, 1])
        >>> children = crossover.crossover(parent1, parent2)
        >>> print(children.shape)  # (2, 6)
    """

    def __init__(self, dev: bool = False):
        """Initialize two-point crossover.

        Args:
            dev: Debug mode (print crossover details)
        """
        super().__init__(dev)

    def crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """Perform two-point crossover.

        Args:
            parent_a: First parent chromosome
            parent_b: Second parent chromosome

        Returns:
            np.ndarray: Two children (shape: 2 × num_features)

        Raises:
            ValueError: If parents have different sizes or size < 3
        """
        if parent_a.size != parent_b.size:
            raise ValueError("Parents must have the same size.")

        if parent_a.size < 3:
            raise ValueError("Parent size must be at least 3 for two-point crossover.")

        # Pick two random crossover points and sort them
        point1, point2 = sorted(np.random.choice(range(1, parent_a.size), size=2, replace=False))

        # Create children by swapping middle segment
        child_1 = np.concatenate([
            parent_a[:point1],
            parent_b[point1:point2],
            parent_a[point2:]
        ])
        child_2 = np.concatenate([
            parent_b[:point1],
            parent_a[point1:point2],
            parent_b[point2:]
        ])

        genes = [child_1, child_2]

        # Debug output
        if self.dev:
            print(f"TwoPointCrossover: Points={point1}, {point2}")
            print(f"Parent A: {parent_a}")
            print(f"Parent B: {parent_b}")
            print(f"Child 1:  {child_1}")
            print(f"Child 2:  {child_2}")
            print("=" * 40)

        return np.array(genes)
