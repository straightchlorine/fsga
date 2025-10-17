"""Edge case tests for FSGA components."""

import numpy as np
import pytest

from fsga.core.genetic_algorithm import GeneticAlgorithm
from fsga.core.population import Population
from fsga.evaluators.accuracy_evaluator import AccuracyEvaluator
from fsga.ml.models import ModelWrapper
from fsga.operators.uniform_crossover import UniformCrossover
from fsga.mutations.bitflip_mutation import BitFlipMutation
from fsga.selectors.tournament_selector import TournamentSelector


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_single_feature_dataset(self):
        """Test GA with dataset having only 1 feature."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = np.array([0, 0, 0, 1, 1, 1])

        X_train, X_test = X[:4], X[4:]
        y_train, y_test = y[:4], y[4:]

        model = ModelWrapper("rf", n_estimators=5, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)
        selector = TournamentSelector(evaluator, tournament_size=2)
        crossover = UniformCrossover()
        mutation = BitFlipMutation(probability=0.1)

        ga = GeneticAlgorithm(
            num_features=1,
            evaluator=evaluator,
            selector=selector,
            crossover_operator=crossover,
            mutation_operator=mutation,
            population_size=5,
            num_generations=3,
            verbose=False,
        )

        results = ga.evolve()
        assert results["best_chromosome"].shape == (1,)
        assert results["best_fitness"] >= 0.0

    def test_very_small_population(self):
        """Test GA with minimum viable population size."""
        X = np.random.rand(20, 4)
        y = np.random.randint(0, 2, size=20)

        X_train, X_test = X[:15], X[15:]
        y_train, y_test = y[:15], y[15:]

        model = ModelWrapper("rf", n_estimators=5, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)
        selector = TournamentSelector(evaluator, tournament_size=2)
        crossover = UniformCrossover()
        mutation = BitFlipMutation(probability=0.1)

        # Minimum population size for tournament_size=2
        ga = GeneticAlgorithm(
            num_features=4,
            evaluator=evaluator,
            selector=selector,
            crossover_operator=crossover,
            mutation_operator=mutation,
            population_size=2,
            num_generations=2,
            verbose=False,
        )

        results = ga.evolve()
        assert len(results["best_chromosome"]) == 4

    def test_single_generation(self):
        """Test GA with only 1 generation."""
        X = np.random.rand(20, 5)
        y = np.random.randint(0, 2, size=20)

        X_train, X_test = X[:15], X[15:]
        y_train, y_test = y[:15], y[15:]

        model = ModelWrapper("rf", n_estimators=5, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)
        selector = TournamentSelector(evaluator, tournament_size=2)
        crossover = UniformCrossover()
        mutation = BitFlipMutation(probability=0.1)

        ga = GeneticAlgorithm(
            num_features=5,
            evaluator=evaluator,
            selector=selector,
            crossover_operator=crossover,
            mutation_operator=mutation,
            population_size=5,
            num_generations=1,
            verbose=False,
        )

        results = ga.evolve()
        assert len(results["best_fitness_history"]) == 1

    def test_zero_mutation_rate(self):
        """Test GA with zero mutation rate."""
        X = np.random.rand(20, 4)
        y = np.random.randint(0, 2, size=20)

        X_train, X_test = X[:15], X[15:]
        y_train, y_test = y[:15], y[15:]

        model = ModelWrapper("rf", n_estimators=5, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)
        selector = TournamentSelector(evaluator, tournament_size=2)
        crossover = UniformCrossover()
        mutation = BitFlipMutation(probability=0.0)  # No mutation

        ga = GeneticAlgorithm(
            num_features=4,
            evaluator=evaluator,
            selector=selector,
            crossover_operator=crossover,
            mutation_operator=mutation,
            population_size=10,
            num_generations=5,
            verbose=False,
        )

        results = ga.evolve()
        assert results["best_fitness"] >= 0.0

    def test_high_dimensional_features(self):
        """Test GA with many features (high-dimensional)."""
        X = np.random.rand(50, 100)  # 100 features
        y = np.random.randint(0, 2, size=50)

        X_train, X_test = X[:40], X[40:]
        y_train, y_test = y[:40], y[40:]

        model = ModelWrapper("rf", n_estimators=5, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)
        selector = TournamentSelector(evaluator, tournament_size=3)
        crossover = UniformCrossover()
        mutation = BitFlipMutation(probability=0.01)

        ga = GeneticAlgorithm(
            num_features=100,
            evaluator=evaluator,
            selector=selector,
            crossover_operator=crossover,
            mutation_operator=mutation,
            population_size=10,
            num_generations=2,
            verbose=False,
        )

        results = ga.evolve()
        assert results["best_chromosome"].shape == (100,)

    def test_imbalanced_dataset(self):
        """Test GA with highly imbalanced classes."""
        # 90% class 0, 10% class 1
        X = np.random.rand(100, 5)
        y = np.array([0] * 90 + [1] * 10)

        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)
        selector = TournamentSelector(evaluator, tournament_size=3)
        crossover = UniformCrossover()
        mutation = BitFlipMutation(probability=0.1)

        ga = GeneticAlgorithm(
            num_features=5,
            evaluator=evaluator,
            selector=selector,
            crossover_operator=crossover,
            mutation_operator=mutation,
            population_size=10,
            num_generations=3,
            verbose=False,
        )

        results = ga.evolve()
        assert results["best_fitness"] >= 0.0

    def test_all_identical_features(self):
        """Test with dataset where all features are identical."""
        # All features have same values
        X = np.ones((20, 5))
        y = np.random.randint(0, 2, size=20)

        X_train, X_test = X[:15], X[15:]
        y_train, y_test = y[:15], y[15:]

        model = ModelWrapper("rf", n_estimators=5, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)
        selector = TournamentSelector(evaluator, tournament_size=2)
        crossover = UniformCrossover()
        mutation = BitFlipMutation(probability=0.1)

        ga = GeneticAlgorithm(
            num_features=5,
            evaluator=evaluator,
            selector=selector,
            crossover_operator=crossover,
            mutation_operator=mutation,
            population_size=5,
            num_generations=2,
            verbose=False,
        )

        # Should not crash
        results = ga.evolve()
        assert results is not None

    def test_early_stopping_on_first_generation(self):
        """Test early stopping that triggers immediately."""
        X = np.random.rand(20, 4)
        y = np.array([0] * 20)  # All same class

        X_train, X_test = X[:15], X[15:]
        y_train, y_test = y[:15], y[15:]

        model = ModelWrapper("rf", n_estimators=5, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)
        selector = TournamentSelector(evaluator, tournament_size=2)
        crossover = UniformCrossover()
        mutation = BitFlipMutation(probability=0.1)

        ga = GeneticAlgorithm(
            num_features=4,
            evaluator=evaluator,
            selector=selector,
            crossover_operator=crossover,
            mutation_operator=mutation,
            population_size=5,
            num_generations=100,
            early_stopping_patience=1,
            verbose=False,
        )

        results = ga.evolve()
        # Should stop early
        assert len(results["best_fitness_history"]) < 100

    def test_mutation_probability_validation(self):
        """Test that invalid mutation probabilities are rejected."""
        with pytest.raises(ValueError):
            BitFlipMutation(probability=-0.1)

        with pytest.raises(ValueError):
            BitFlipMutation(probability=1.5)

    def test_tournament_size_validation(self, simple_evaluator):
        """Test tournament size validation during selection."""
        # Tournament size can be set to any value but validated during select
        selector = TournamentSelector(simple_evaluator, tournament_size=100)

        # Set small population
        small_pop = np.array([[1, 0], [0, 1]])
        selector.population = small_pop

        # Should fail when trying to select (tournament > population)
        with pytest.raises(ValueError, match="cannot be larger"):
            selector.select()

    def test_population_with_duplicate_chromosomes(self, simple_evaluator):
        """Test population with many duplicate chromosomes."""
        selector = TournamentSelector(simple_evaluator, tournament_size=2)
        pop = Population(num_features=4, population_size=10, selector=selector)

        # Create population with many duplicates
        pop.chromosomes = np.array([
            [1, 0, 1, 0],
            [1, 0, 1, 0],  # Duplicate
            [1, 0, 1, 0],  # Duplicate
            [0, 1, 0, 1],
            [0, 1, 0, 1],  # Duplicate
            [1, 1, 0, 0],
            [1, 1, 0, 0],  # Duplicate
            [0, 0, 1, 1],
            [1, 0, 1, 0],  # Duplicate
            [0, 1, 0, 1],  # Duplicate
        ])

        diversity = pop.measure_diversity()
        # Only 4 unique chromosomes out of 10
        assert diversity == 40.0

    def test_very_large_tournament_size(self, simple_evaluator):
        """Test tournament size equal to population size."""
        selector = TournamentSelector(simple_evaluator, tournament_size=10)
        population = np.random.randint(0, 2, size=(10, 5))
        selector.population = population

        parent1, parent2 = selector.select()

        # Should still work, just selects best from entire population
        assert parent1.shape == (5,)
        assert parent2.shape == (5,)

    def test_sparse_population_initialization(self, simple_evaluator):
        """Test very sparse population (almost all zeros)."""
        selector = TournamentSelector(simple_evaluator, tournament_size=2)
        pop = Population(num_features=100, population_size=20, selector=selector)

        pop.initialize_population(strategy="sparse", sparsity=0.01)

        # Most genes should be 0, but at least 1 per chromosome
        assert np.all(pop.chromosomes.sum(axis=1) >= 1)
        # Overall density should be low
        assert pop.chromosomes.mean() < 0.1

    def test_dense_population_initialization(self, simple_evaluator):
        """Test very dense population (almost all ones)."""
        selector = TournamentSelector(simple_evaluator, tournament_size=2)
        pop = Population(num_features=100, population_size=20, selector=selector)

        pop.initialize_population(strategy="dense", density=0.99)

        # Most genes should be 1
        assert pop.chromosomes.mean() > 0.9

    def test_crossover_with_identical_parents(self):
        """Test that crossover with identical parents produces identical children."""
        crossover = UniformCrossover()
        parent = np.array([1, 0, 1, 0, 1])

        children = crossover.crossover(parent, parent)

        # Both children should equal parent
        assert np.array_equal(children[0], parent)
        assert np.array_equal(children[1], parent)

    def test_evaluator_with_single_sample(self):
        """Test evaluator with minimal training data."""
        # Just 2 samples (minimum for binary classification)
        X_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([0, 1])
        X_test = np.array([[2, 3]])
        y_test = np.array([0])

        model = ModelWrapper("rf", n_estimators=3, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

        chromosome = np.array([1, 1])
        fitness = evaluator.evaluate(chromosome)

        # Should work without crashing
        assert 0.0 <= fitness <= 1.0
