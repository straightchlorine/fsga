"""Uniform crossover operator.

Ported directly from knapsack-problem/knapsack/operators/uniform_crossover.py
No changes needed - works identically for feature selection.
"""

import numpy as np

from fsga.operators.crossover import Crossover


class UniformCrossover(Crossover):
    """Uniform crossover: each gene randomly chosen from either parent.

    For each gene position, flips a coin to decide which parent contributes
    that gene to each child.

    Example:
        Parent 1: [1, 1, 0, 1, 0]
        Parent 2: [0, 0, 1, 1, 0]
        Mask:     [T, T, F, T, T]  (random boolean array)

        Child 1:  [1, 1, 1, 1, 0]  (T → parent1, F → parent2)
        Child 2:  [0, 0, 0, 1, 0]  (T → parent2, F → parent1)

    Properties:
        - High exploration (genes mixed randomly)
        - Preserves no structure from parents
        - Good for avoiding premature convergence

    Usage:
        >>> crossover = UniformCrossover()
        >>> parent1 = np.array([1, 0, 1, 0, 1])
        >>> parent2 = np.array([0, 1, 1, 1, 0])
        >>> children = crossover.crossover(parent1, parent2)
        >>> print(children.shape)  # (2, 5)
    """

    def __init__(self, dev: bool = False):
        """Initialize uniform crossover.

        Args:
            dev: Debug mode (print crossover details)
        """
        super().__init__(dev)

    def crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """Perform uniform crossover.

        Args:
            parent_a: First parent chromosome
            parent_b: Second parent chromosome

        Returns:
            np.ndarray: Two children (shape: 2 × num_features)
        """
        # Generate random binary mask
        mask = np.random.randint(0, 2, size=parent_a.size).astype(bool)

        # Create children using mask
        # Child 1: Takes genes from parent_a where mask=True, parent_b where False
        # Child 2: Takes genes from parent_b where mask=True, parent_a where False
        child1 = np.where(mask, parent_a, parent_b)
        child2 = np.where(mask, parent_b, parent_a)

        genes = [child1, child2]

        # Debug output
        if self.dev:
            print(f"Crossover on {parent_a} and {parent_b}")
            print(f"Generated mask: {mask}")
            print(f"Generated children: {genes[0]} and {genes[1]}")
            print("=" * 20)

        return np.array(genes)
