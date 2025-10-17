"""Multi-point crossover operator for feature selection.

Adapted from knapsack-problem/knapsack/operators/multi_point_crossover.py
Key changes:
- Updated imports to fsga modules
- Maintained binary chromosome compatibility
- Enhanced docstring for feature selection context
"""

import logging

import numpy as np

from fsga.operators.crossover import Crossover


class MultiPointCrossover(Crossover):
    """Multi-point crossover operator for binary chromosomes.

    Randomly selects multiple crossover points and alternates swapping segments
    between parents to create offspring. This allows more diversity compared to
    single-point crossover.

    Attributes:
        points: Fixed crossover points (optional). If None, randomly selects 1-4 points
        dev: Enable debug logging (default: False)

    Example:
        >>> crossover = MultiPointCrossover(points=[2, 5])
        >>> parent1 = np.array([1, 1, 0, 0, 1, 1, 0])
        >>> parent2 = np.array([0, 0, 1, 1, 0, 0, 1])
        >>> children = crossover.crossover(parent1, parent2)
        >>> # Swaps segments: [1,1] | [0,0,1] | [1,0] -> children
    """

    def __init__(self, points=None, dev=False):
        """Initialize multi-point crossover.

        Args:
            points: List of fixed crossover points (1-indexed). If None, randomly selects
            dev: Enable debug logging (default: False)
        """
        self.points = points
        self.dev = dev
        self._last_crossover_points = None

    def _validate_points(self, parent):
        """Validate and generate crossover points.

        Args:
            parent: Parent chromosome to determine valid point range

        Returns:
            np.ndarray: Sorted array of crossover points

        Raises:
            ValueError: If parent size < 2 or points are out of range
        """
        if parent.size < 2:
            raise ValueError("Parent size must be at least 2.")
        if self._last_crossover_points is not None:
            logging.info("Using cached crossover points")
            return self._last_crossover_points

        if self.points is not None:
            if not all(1 <= p < parent.size for p in self.points):
                raise ValueError("All points must be between 1 and parent.size - 1.")
            points = np.sort(self.points)
        else:
            # Random: select 1-4 points (capped at parent.size - 1)
            num_points = np.random.randint(1, min(5, parent.size - 1))
            points = np.sort(
                np.random.choice(range(1, parent.size), size=num_points, replace=False)
            )
        return points

    def crossover(self, parent_a, parent_b):
        """Perform multi-point crossover on two parents.

        Args:
            parent_a: First parent chromosome
            parent_b: Second parent chromosome

        Returns:
            np.ndarray: Two offspring chromosomes (shape: [2, num_features])

        Raises:
            ValueError: If parents have different sizes
        """
        if parent_a.size != parent_b.size:
            raise ValueError("Parents must have the same size.")

        points = self._validate_points(parent_a)
        logging.info(f"MultiPointCrossover: Using crossover points {points}")

        children = [parent_a.copy(), parent_b.copy()]

        # Alternate swapping segments between crossover points
        for i, start in enumerate(points):
            end = points[i + 1] if i + 1 < len(points) else None
            children[i % 2][start:end], children[(i + 1) % 2][start:end] = (
                children[(i + 1) % 2][start:end].copy(),
                children[i % 2][start:end].copy(),
            )

        self._last_crossover_points = points
        if self.dev:
            logging.info(f"Crossover on {parent_a} and {parent_b}")
            logging.info(f"Generated children: {children[0]} and {children[1]}")

        return np.array(children)
