"""Population management for genetic algorithm.

Adapted from knapsack-problem/knapsack/population.py
Key changes:
- Removed dataset-specific initialization (value_biased, weight_constrained, etc.)
- Kept generic strategies (random, uniform, normal)
- Added placeholder for feature-aware initialization (to be implemented later)
- Simplified interface (no dataset dependency)
"""

import numpy as np

from fsga.selectors.selector import Selector


class Population:
    """Population of chromosomes for feature selection.

    A chromosome is a binary array where chromosome[i]=1 means "include feature i".

    Attributes:
        num_features: Length of each chromosome (number of features)
        population_size: Number of chromosomes in population
        selector: Parent selection strategy
        chromosomes: Numpy array of chromosomes (shape: population_size × num_features)

    Example:
        >>> pop = Population(num_features=30, population_size=50, selector=selector)
        >>> pop.initialize_population(strategy='random')
        >>> diversity = pop.measure_diversity()
    """

    def __init__(
        self,
        num_features: int,
        population_size: int,
        selector: Selector,
    ):
        """Initialize population.

        Args:
            num_features: Number of features (chromosome length)
            population_size: Number of chromosomes
            selector: Parent selection strategy
        """
        self.num_features = num_features
        self.population_size = population_size
        self.selector = selector

        # Empty chromosome array (initialized by initialize_population())
        self.chromosomes: np.ndarray = np.array([])

    def update_selector(self):
        """Update selector with current population."""
        self.selector.population = self.chromosomes

    def _gen_genes_random(self) -> np.ndarray:
        """Generate random binary chromosome (50% chance for 0 or 1)."""
        return np.random.randint(0, 2, size=self.num_features)

    def _gen_genes_uniform(self) -> np.ndarray:
        """Generate chromosome using uniform distribution.

        Alias for _gen_genes_random for compatibility.
        """
        return np.random.choice([0, 1], size=self.num_features)

    def _gen_genes_normal(self) -> np.ndarray:
        """Generate chromosome using normal distribution.

        Uses Gaussian distribution centered at 0.5 (stddev=0.15),
        then thresholds at 0.5 to create binary chromosome.
        Results in ~50% features selected, but with clustering.
        """
        probabilities = np.random.normal(0.5, 0.15, self.num_features)
        probabilities = np.clip(probabilities, 0, 1)
        return (probabilities > 0.5).astype(int)

    def _gen_genes_sparse(self, sparsity: float = 0.2) -> np.ndarray:
        """Generate sparse chromosome (few features selected).

        Args:
            sparsity: Probability of selecting each feature (default: 0.2 = 20%)

        Returns:
            Binary chromosome with ~sparsity% features selected
        """
        return (np.random.random(self.num_features) < sparsity).astype(int)

    def _gen_genes_dense(self, density: float = 0.8) -> np.ndarray:
        """Generate dense chromosome (many features selected).

        Args:
            density: Probability of selecting each feature (default: 0.8 = 80%)

        Returns:
            Binary chromosome with ~density% features selected
        """
        return (np.random.random(self.num_features) < density).astype(int)

    def initialize_population(self, strategy: str = "random", **kwargs):
        """Initialize population using specified strategy.

        Args:
            strategy: Initialization strategy. Options:
                - "random": Uniform random (50% probability per feature)
                - "uniform": Alias for random
                - "normal": Gaussian distribution around 50%
                - "sparse": Few features selected (~20%)
                - "dense": Many features selected (~80%)
            **kwargs: Additional arguments for specific strategies
                - sparsity: For "sparse" strategy (default: 0.2)
                - density: For "dense" strategy (default: 0.8)

        Raises:
            ValueError: If unknown strategy specified

        Example:
            >>> pop.initialize_population(strategy='sparse', sparsity=0.1)
        """
        strategies = {
            "random": self._gen_genes_random,
            "uniform": self._gen_genes_uniform,
            "normal": self._gen_genes_normal,
            "sparse": lambda: self._gen_genes_sparse(kwargs.get("sparsity", 0.2)),
            "dense": lambda: self._gen_genes_dense(kwargs.get("density", 0.8)),
        }

        if strategy not in strategies:
            raise ValueError(
                f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}"
            )

        # Generate population
        self.chromosomes = np.array(
            [strategies[strategy]() for _ in range(self.population_size)]
        )

        # Ensure at least one feature is selected in each chromosome
        for i, chrom in enumerate(self.chromosomes):
            if chrom.sum() == 0:  # No features selected
                # Randomly select one feature
                random_idx = np.random.randint(0, self.num_features)
                self.chromosomes[i][random_idx] = 1

        # Update selector with new population
        self.update_selector()

    def select_parents(self) -> tuple[np.ndarray, np.ndarray]:
        """Select two parents for reproduction.

        Returns:
            tuple: Two parent chromosomes (numpy arrays)
        """
        return self.selector.select()

    def measure_diversity(self) -> float:
        """Calculate genetic diversity in population.

        Diversity = (number of unique chromosomes) / (population size) × 100

        Returns:
            float: Diversity percentage (0-100)
                - 100 = all chromosomes are unique
                - 0 = all chromosomes are identical

        Example:
            >>> diversity = pop.measure_diversity()
            >>> print(f"Diversity: {diversity:.2f}%")
        """
        unique = {tuple(c) for c in self.chromosomes}
        return len(unique) / len(self.chromosomes) * 100

    def add_chromosome(self, chromosomes: np.ndarray):
        """Add chromosome(s) to the population.

        Args:
            chromosomes: Single chromosome (1D array) or multiple (2D array)

        Returns:
            int: Index of last added chromosome

        Example:
            >>> child = np.array([1, 0, 1, 0])
            >>> pop.add_chromosome(child)
            >>>
            >>> # Or add multiple
            >>> children = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
            >>> pop.add_chromosome(children)
        """
        # Handle both single chromosome (1D) and multiple (2D)
        if chromosomes.ndim == 1:
            chromosomes = chromosomes.reshape(1, -1)

        # Initialize if empty
        if not hasattr(self, "chromosomes") or self.chromosomes.size == 0:
            self.chromosomes = np.empty(
                (0, chromosomes.shape[1]), dtype=chromosomes.dtype
            )

        # Add to population
        self.chromosomes = np.vstack([self.chromosomes, chromosomes])

        return len(self.chromosomes) - 1

    def get_statistics(self) -> dict:
        """Get population statistics.

        Returns:
            dict: Statistics including:
                - size: Population size
                - diversity: Genetic diversity (%)
                - avg_features: Average features selected per chromosome
                - min_features: Minimum features in any chromosome
                - max_features: Maximum features in any chromosome
        """
        features_selected = self.chromosomes.sum(axis=1)

        return {
            "size": len(self.chromosomes),
            "diversity": self.measure_diversity(),
            "avg_features": float(features_selected.mean()),
            "min_features": int(features_selected.min()),
            "max_features": int(features_selected.max()),
        }
