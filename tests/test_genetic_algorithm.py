"""Unit tests for GeneticAlgorithm class."""

import numpy as np
import pytest

from fsga.core.genetic_algorithm import GeneticAlgorithm, Timer


class TestTimer:
    """Test suite for Timer context manager."""

    def test_timer_measures_time(self):
        """Test that Timer correctly measures elapsed time."""
        import time

        with Timer() as timer:
            time.sleep(0.01)  # Sleep for 10ms

        assert timer.interval >= 10.0  # At least 10ms
        assert timer.interval < 100.0  # Less than 100ms (sanity check)


class TestGeneticAlgorithm:
    """Test suite for GeneticAlgorithm class."""

    @pytest.fixture
    def ga_components(self, iris_data_split):
        """Create GA components for testing."""
        from fsga.evaluators.accuracy_evaluator import AccuracyEvaluator
        from fsga.ml.models import ModelWrapper
        from fsga.operators.uniform_crossover import UniformCrossover
        from fsga.mutations.bitflip_mutation import BitFlipMutation
        from fsga.selectors.tournament_selector import TournamentSelector

        X_train, X_test, y_train, y_test = iris_data_split
        model = ModelWrapper("rf", n_estimators=10, random_state=42)
        evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)
        selector = TournamentSelector(evaluator, tournament_size=3)
        crossover = UniformCrossover()
        mutation = BitFlipMutation(probability=0.1)

        return {
            "num_features": X_train.shape[1],
            "evaluator": evaluator,
            "selector": selector,
            "crossover": crossover,
            "mutation": mutation,
        }

    def test_initialization(self, ga_components):
        """Test GA initialization."""
        ga = GeneticAlgorithm(
            num_features=ga_components["num_features"],
            evaluator=ga_components["evaluator"],
            selector=ga_components["selector"],
            crossover_operator=ga_components["crossover"],
            mutation_operator=ga_components["mutation"],
            population_size=20,
            num_generations=5,
            mutation_rate=0.1,
            verbose=False,
        )

        assert ga.num_features == ga_components["num_features"]
        assert ga.population_size == 20
        assert ga.generations == 5
        assert ga.mutation_rate == 0.1
        assert ga.verbose is False
        assert len(ga.population.chromosomes) == 20

    def test_property_setters(self, ga_components):
        """Test property setters."""
        ga = GeneticAlgorithm(
            num_features=ga_components["num_features"],
            evaluator=ga_components["evaluator"],
            selector=ga_components["selector"],
            crossover_operator=ga_components["crossover"],
            mutation_operator=ga_components["mutation"],
            population_size=10,
        )

        # Test setters
        ga.population_size = 20
        assert ga.population_size == 20

        ga.generations = 50
        assert ga.generations == 50

        ga.mutation_rate = 0.05
        assert ga.mutation_rate == 0.05
        assert ga.mutation_operator.probability == 0.05

    def test_clear_metrics(self, ga_components):
        """Test clearing metrics."""
        ga = GeneticAlgorithm(
            num_features=ga_components["num_features"],
            evaluator=ga_components["evaluator"],
            selector=ga_components["selector"],
            crossover_operator=ga_components["crossover"],
            mutation_operator=ga_components["mutation"],
            population_size=10,
        )

        # Add some dummy metrics
        ga.best_fitness = [0.8, 0.9, 0.95]
        ga.average_fitness = [0.7, 0.75, 0.8]
        ga.optimal_generation = 2

        ga.clear_metrics()

        assert ga.best_fitness == []
        assert ga.average_fitness == []
        assert ga.worst_fitness == []
        assert ga.diversity == []
        assert ga.optimal_generation == 0

    def test_reinitialize_population(self, ga_components):
        """Test population reinitialization."""
        ga = GeneticAlgorithm(
            num_features=ga_components["num_features"],
            evaluator=ga_components["evaluator"],
            selector=ga_components["selector"],
            crossover_operator=ga_components["crossover"],
            mutation_operator=ga_components["mutation"],
            population_size=10,
        )

        old_population = ga.population.chromosomes.copy()
        ga.reinitialize_population(strategy="sparse")
        new_population = ga.population.chromosomes

        # Population should be different after reinitialization
        assert not np.array_equal(old_population, new_population)
        assert len(new_population) == 10

    def test_evolve_returns_correct_structure(self, ga_components):
        """Test that evolve() returns correct result structure."""
        ga = GeneticAlgorithm(
            num_features=ga_components["num_features"],
            evaluator=ga_components["evaluator"],
            selector=ga_components["selector"],
            crossover_operator=ga_components["crossover"],
            mutation_operator=ga_components["mutation"],
            population_size=10,
            num_generations=3,
            verbose=False,
        )

        results = ga.evolve()

        # Check all required keys are present
        assert "best_chromosome" in results
        assert "best_fitness" in results
        assert "best_fitness_history" in results
        assert "average_fitness_history" in results
        assert "worst_fitness_history" in results
        assert "diversity_history" in results
        assert "optimal_generation" in results
        assert "execution_time_ms" in results
        assert "converged" in results

        # Check types and shapes
        assert isinstance(results["best_chromosome"], np.ndarray)
        assert results["best_chromosome"].shape[0] == ga_components["num_features"]
        assert isinstance(results["best_fitness"], float)
        assert len(results["best_fitness_history"]) == 3
        assert results["execution_time_ms"] > 0

    def test_evolve_improves_fitness(self, ga_components):
        """Test that evolution improves fitness over generations."""
        ga = GeneticAlgorithm(
            num_features=ga_components["num_features"],
            evaluator=ga_components["evaluator"],
            selector=ga_components["selector"],
            crossover_operator=ga_components["crossover"],
            mutation_operator=ga_components["mutation"],
            population_size=20,
            num_generations=10,
            verbose=False,
        )

        results = ga.evolve()

        # Final best fitness should be >= initial best fitness
        initial_fitness = results["best_fitness_history"][0]
        final_fitness = results["best_fitness"]
        assert final_fitness >= initial_fitness

    def test_early_stopping(self, ga_components):
        """Test early stopping functionality."""
        ga = GeneticAlgorithm(
            num_features=ga_components["num_features"],
            evaluator=ga_components["evaluator"],
            selector=ga_components["selector"],
            crossover_operator=ga_components["crossover"],
            mutation_operator=ga_components["mutation"],
            population_size=10,
            num_generations=100,  # Large number
            early_stopping_patience=3,
            verbose=False,
        )

        results = ga.evolve()

        # Should stop early (not run all 100 generations)
        assert len(results["best_fitness_history"]) < 100
        assert results["converged"] is True

    def test_no_early_stopping_without_patience(self, ga_components):
        """Test that evolution runs full generations without early stopping."""
        ga = GeneticAlgorithm(
            num_features=ga_components["num_features"],
            evaluator=ga_components["evaluator"],
            selector=ga_components["selector"],
            crossover_operator=ga_components["crossover"],
            mutation_operator=ga_components["mutation"],
            population_size=10,
            num_generations=5,
            early_stopping_patience=None,
            verbose=False,
        )

        results = ga.evolve()

        # Should run all 5 generations
        assert len(results["best_fitness_history"]) == 5
        assert results["converged"] is False

    def test_new_generation_maintains_population_size(self, ga_components):
        """Test that new_generation maintains population size."""
        ga = GeneticAlgorithm(
            num_features=ga_components["num_features"],
            evaluator=ga_components["evaluator"],
            selector=ga_components["selector"],
            crossover_operator=ga_components["crossover"],
            mutation_operator=ga_components["mutation"],
            population_size=20,
        )

        new_pop = ga.new_generation(current_generation=0)

        assert len(new_pop.chromosomes) == 20
        assert new_pop.chromosomes.shape[1] == ga_components["num_features"]

    def test_get_best_solution_single(self, ga_components):
        """Test getting single best solution."""
        ga = GeneticAlgorithm(
            num_features=ga_components["num_features"],
            evaluator=ga_components["evaluator"],
            selector=ga_components["selector"],
            crossover_operator=ga_components["crossover"],
            mutation_operator=ga_components["mutation"],
            population_size=10,
        )

        best = ga.get_best_solution(n=1)

        assert len(best) == 1
        assert isinstance(best[0], np.ndarray)
        assert best[0].shape[0] == ga_components["num_features"]

    def test_get_best_solution_multiple(self, ga_components):
        """Test getting top N solutions."""
        ga = GeneticAlgorithm(
            num_features=ga_components["num_features"],
            evaluator=ga_components["evaluator"],
            selector=ga_components["selector"],
            crossover_operator=ga_components["crossover"],
            mutation_operator=ga_components["mutation"],
            population_size=10,
        )

        best = ga.get_best_solution(n=3)

        assert len(best) == 3
        # Verify they are sorted by fitness (descending)
        fitness_scores = [ga_components["evaluator"].evaluate(chrom) for chrom in best]
        assert fitness_scores == sorted(fitness_scores, reverse=True)

    def test_diversity_tracking(self, ga_components):
        """Test that diversity is tracked across generations."""
        ga = GeneticAlgorithm(
            num_features=ga_components["num_features"],
            evaluator=ga_components["evaluator"],
            selector=ga_components["selector"],
            crossover_operator=ga_components["crossover"],
            mutation_operator=ga_components["mutation"],
            population_size=10,
            num_generations=5,
            verbose=False,
        )

        results = ga.evolve()

        # Should have diversity measurements for each generation
        assert len(results["diversity_history"]) == 5
        # All diversity values should be between 0 and 100
        assert all(0 <= d <= 100 for d in results["diversity_history"])

    def test_optimal_generation_tracking(self, ga_components):
        """Test that optimal generation is correctly tracked."""
        ga = GeneticAlgorithm(
            num_features=ga_components["num_features"],
            evaluator=ga_components["evaluator"],
            selector=ga_components["selector"],
            crossover_operator=ga_components["crossover"],
            mutation_operator=ga_components["mutation"],
            population_size=10,
            num_generations=5,
            verbose=False,
        )

        results = ga.evolve()

        # Optimal generation should be within valid range
        assert 0 <= results["optimal_generation"] < 5
        # Best fitness should match the fitness at optimal generation
        optimal_fitness = results["best_fitness_history"][results["optimal_generation"]]
        assert optimal_fitness == results["best_fitness"]

    def test_verbose_output(self, ga_components, capsys):
        """Test verbose output during evolution."""
        ga = GeneticAlgorithm(
            num_features=ga_components["num_features"],
            evaluator=ga_components["evaluator"],
            selector=ga_components["selector"],
            crossover_operator=ga_components["crossover"],
            mutation_operator=ga_components["mutation"],
            population_size=5,
            num_generations=2,
            verbose=True,
        )

        ga.evolve()
        captured = capsys.readouterr()

        # Should have printed generation info
        assert "Generation 1/" in captured.out
        assert "Best=" in captured.out
        assert "Evolution completed" in captured.out

    def test_fitness_history_consistency(self, ga_components):
        """Test that fitness histories are consistent."""
        ga = GeneticAlgorithm(
            num_features=ga_components["num_features"],
            evaluator=ga_components["evaluator"],
            selector=ga_components["selector"],
            crossover_operator=ga_components["crossover"],
            mutation_operator=ga_components["mutation"],
            population_size=10,
            num_generations=5,
            verbose=False,
        )

        results = ga.evolve()

        # All histories should have same length
        assert len(results["best_fitness_history"]) == len(results["average_fitness_history"])
        assert len(results["best_fitness_history"]) == len(results["worst_fitness_history"])
        assert len(results["best_fitness_history"]) == len(results["diversity_history"])

        # Best >= Average >= Worst for each generation
        for i in range(len(results["best_fitness_history"])):
            assert results["best_fitness_history"][i] >= results["average_fitness_history"][i]
            assert results["average_fitness_history"][i] >= results["worst_fitness_history"][i]
