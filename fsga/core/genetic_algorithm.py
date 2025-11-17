"""Genetic Algorithm engine for feature selection.

Adapted from knapsack-problem/knapsack/genetic_algorithm.py
Key changes:
- Removed dataset-specific logic (weights, capacity)
- Added support for ML model training integration via evaluators
- Added early stopping capability
- Enhanced result dictionary with more metrics
- Added logging support
"""

import time

import numpy as np

from fsga.core.population import Population
from fsga.evaluators.evaluator import Evaluator
from fsga.mutations.mutation import Mutation
from fsga.operators.crossover import Crossover
from fsga.selectors.selector import Selector


class Timer:
    """Context manager for timing code execution."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = (self.end - self.start) * 1000  # milliseconds


class GeneticAlgorithm:
    """Genetic Algorithm for feature selection optimization.

    Uses evolutionary strategies to find optimal feature subsets that maximize
    ML model performance.

    Attributes:
        num_features: Number of features in the dataset
        evaluator: Fitness evaluator (trains ML model)
        selector: Parent selection strategy
        crossover_operator: Crossover operator for recombination
        mutation_operator: Mutation operator for variation
        population_size: Number of chromosomes in population
        num_generations: Maximum generations to evolve
        mutation_rate: Probability of mutation per gene
        early_stopping_patience: Stop if no improvement for N generations

    Example:
        >>> from fsga.core.genetic_algorithm import GeneticAlgorithm
        >>> from fsga.evaluators.accuracy_evaluator import AccuracyEvaluator
        >>>
        >>> ga = GeneticAlgorithm(
        ...     num_features=30,
        ...     evaluator=evaluator,
        ...     selector=selector,
        ...     crossover_operator=crossover,
        ...     mutation_operator=mutation,
        ...     population_size=50,
        ...     num_generations=100
        ... )
        >>> results = ga.evolve()
        >>> print(f"Best fitness: {results['best_fitness']}")
    """

    def __init__(
        self,
        num_features: int,
        evaluator: Evaluator,
        selector: Selector,
        crossover_operator: Crossover,
        mutation_operator: Mutation,
        population_size: int = 50,
        num_generations: int = 100,
        mutation_rate: float = 0.01,
        early_stopping_patience: int | None = None,
        verbose: bool = False,
    ):
        """Initialize Genetic Algorithm.

        Args:
            num_features: Number of features in dataset (chromosome length)
            evaluator: Fitness evaluator
            selector: Parent selection strategy
            crossover_operator: Crossover operator
            mutation_operator: Mutation operator
            population_size: Number of chromosomes (default: 50)
            num_generations: Maximum generations (default: 100)
            mutation_rate: Mutation probability (default: 0.01)
            early_stopping_patience: Stop after N gens without improvement (default: None)
            verbose: Print progress (default: False)
        """
        self._evaluator = evaluator
        self._selector = selector
        self._selector.evaluator = evaluator

        self.num_features = num_features
        self._population_size = population_size
        self._max_generations = num_generations
        self._mutation_rate = mutation_rate
        self._early_stopping_patience = early_stopping_patience
        self.verbose = verbose

        self._mutation_operator = mutation_operator
        if hasattr(self._mutation_operator, "probability"):
            self._mutation_operator.probability = mutation_rate

        self._crossover_operator = crossover_operator

        # Initialize population
        self.population = Population(
            num_features=self.num_features,
            population_size=self.population_size,
            selector=self._selector,
        )
        self.population.initialize_population(strategy="random")

        # Metrics storage
        self.best_fitness = []
        self.average_fitness = []
        self.worst_fitness = []
        self.diversity = []
        self.optimal_generation = 0

    @property
    def selector(self) -> Selector:
        return self._selector

    @selector.setter
    def selector(self, selector: Selector):
        self._selector = selector
        self._selector.evaluator = self.evaluator
        self.population.selector = selector
        self.population.update_selector()

    @property
    def crossover_operator(self) -> Crossover:
        return self._crossover_operator

    @crossover_operator.setter
    def crossover_operator(self, operator: Crossover):
        self._crossover_operator = operator

    @property
    def population_size(self) -> int:
        return self._population_size

    @population_size.setter
    def population_size(self, size: int):
        self._population_size = size

    @property
    def evaluator(self) -> Evaluator:
        return self._evaluator

    @evaluator.setter
    def evaluator(self, evaluator: Evaluator):
        self._evaluator = evaluator

    @property
    def generations(self) -> int:
        return self._max_generations

    @generations.setter
    def generations(self, value: int):
        self._max_generations = value

    @property
    def mutation_rate(self) -> float:
        return self._mutation_rate

    @mutation_rate.setter
    def mutation_rate(self, value: float):
        self._mutation_rate = value
        if self._mutation_operator and hasattr(self._mutation_operator, "probability"):
            self._mutation_operator.probability = value

    @property
    def mutation_operator(self) -> Mutation:
        return self._mutation_operator

    @mutation_operator.setter
    def mutation_operator(self, operator: Mutation):
        self._mutation_operator = operator
        if hasattr(self._mutation_operator, "probability"):
            self._mutation_operator.probability = self._mutation_rate

    def clear_metrics(self):
        """Clear all stored metrics."""
        self.best_fitness = []
        self.average_fitness = []
        self.worst_fitness = []
        self.diversity = []
        self.optimal_generation = 0

    def reinitialize_population(self, strategy: str = "random"):
        """Reinitialize population with given strategy.

        Args:
            strategy: Initialization strategy ("random", "uniform", etc.)
        """
        self.population = Population(
            num_features=self.num_features,
            population_size=self._population_size,
            selector=self._selector,
        )
        self.population.initialize_population(strategy)

    def evolve(self) -> dict:
        """Evolve the population for specified number of generations.

        Returns:
            dict: Results containing:
                - best_chromosome: Best solution found
                - best_fitness: Best fitness value
                - best_fitness_history: Best fitness per generation
                - average_fitness_history: Average fitness per generation
                - worst_fitness_history: Worst fitness per generation
                - diversity_history: Population diversity per generation
                - optimal_generation: When best solution was found
                - execution_time_ms: Total runtime in milliseconds
                - converged: Whether early stopping occurred
        """
        self.clear_metrics()

        no_improvement_count = 0
        best_chromosome = None

        with Timer() as timer:
            for generation in range(self.generations):
                # Evaluate fitness for all chromosomes
                fitness_scores = [
                    self.evaluator.evaluate(chrom)
                    for chrom in self.population.chromosomes
                ]

                # Track metrics
                best_fitness = max(fitness_scores)
                avg_fitness = np.mean(fitness_scores)
                worst_fitness = min(fitness_scores)
                diversity = self.population.measure_diversity()

                # Track best solution
                if not self.best_fitness or best_fitness > max(self.best_fitness):
                    self.optimal_generation = generation
                    best_idx = np.argmax(fitness_scores)
                    best_chromosome = self.population.chromosomes[best_idx].copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                self.best_fitness.append(best_fitness)
                self.average_fitness.append(avg_fitness)
                self.worst_fitness.append(worst_fitness)
                self.diversity.append(diversity)

                if self.verbose:
                    print(
                        f"Generation {generation + 1}/{self.generations}: "
                        f"Best={best_fitness:.4f}, "
                        f"Avg={avg_fitness:.4f}, "
                        f"Diversity={diversity:.2f}%"
                    )

                # Early stopping check
                if (
                    self._early_stopping_patience is not None
                    and no_improvement_count >= self._early_stopping_patience
                ):
                    if self.verbose:
                        print(
                            f"Early stopping at generation {generation + 1} "
                            f"(no improvement for {no_improvement_count} generations)"
                        )
                    break

                # Create next generation
                self.population = self.new_generation(generation)

        if self.verbose:
            print(f"Evolution completed in {timer.interval:.2f} ms")

        # Return comprehensive results
        return {
            "best_chromosome": best_chromosome,
            "best_fitness": max(self.best_fitness),
            "best_fitness_history": self.best_fitness,
            "average_fitness_history": self.average_fitness,
            "worst_fitness_history": self.worst_fitness,
            "diversity_history": self.diversity,
            "optimal_generation": self.optimal_generation,
            "execution_time_ms": timer.interval,
            "converged": no_improvement_count >= (self._early_stopping_patience or float("inf")),
        }

    def new_generation(self, current_generation: int) -> Population:
        """Create a new generation through selection, crossover, and mutation.

        Args:
            current_generation: Current generation number (for dynamic mutation)

        Returns:
            Population: New population of chromosomes
        """
        new_population = Population(
            num_features=self.num_features,
            population_size=self._population_size,
            selector=self._selector,
        )

        while len(new_population.chromosomes) < self.population_size:
            # Selection
            parent1, parent2 = self.population.select_parents()

            # Crossover
            children = self._crossover_operator.crossover(parent1, parent2)

            # Mutation (handle dynamic mutation that needs generation number)
            if hasattr(self._mutation_operator, "max_generations"):
                children = self._mutation_operator.mutate(children, current_generation)
            else:
                children = self._mutation_operator.mutate(children)

            # Add to new population
            new_population.add_chromosome(children)

        return new_population

    def get_best_solution(self, n: int = 1) -> list:
        """Get the best N solutions from current population.

        Args:
            n: Number of top solutions to return (default: 1)

        Returns:
            list: Top N chromosomes sorted by fitness (descending)
        """
        return sorted(
            self.population.chromosomes,
            key=lambda chrom: self.evaluator.evaluate(chrom),
            reverse=True,
        )[:n]
