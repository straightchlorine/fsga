"""Unit tests for Population class."""

import numpy as np
import pytest

from fsga.core.population import Population


class TestPopulation:
    """Test suite for Population class."""

    def test_initialization(self, simple_evaluator):
        """Test population initialization."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=10, population_size=20, selector=selector)

        assert pop.num_features == 10
        assert pop.population_size == 20
        assert pop.selector is selector
        assert pop.chromosomes.size == 0  # Not initialized yet

    def test_initialize_random_strategy(self, simple_evaluator, random_seed):
        """Test random initialization strategy."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=10, population_size=20, selector=selector)
        pop.initialize_population(strategy="random")

        assert pop.chromosomes.shape == (20, 10)
        assert np.all((pop.chromosomes == 0) | (pop.chromosomes == 1))
        # All chromosomes should have at least one feature
        assert np.all(pop.chromosomes.sum(axis=1) > 0)

    def test_initialize_uniform_strategy(self, simple_evaluator, random_seed):
        """Test uniform initialization strategy."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=10, population_size=20, selector=selector)
        pop.initialize_population(strategy="uniform")

        assert pop.chromosomes.shape == (20, 10)
        assert np.all((pop.chromosomes == 0) | (pop.chromosomes == 1))

    def test_initialize_normal_strategy(self, simple_evaluator, random_seed):
        """Test normal distribution initialization strategy."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=100, population_size=50, selector=selector)
        pop.initialize_population(strategy="normal")

        assert pop.chromosomes.shape == (50, 100)
        assert np.all((pop.chromosomes == 0) | (pop.chromosomes == 1))
        # Should be roughly 50% features selected (normal around 0.5)
        avg_selected = pop.chromosomes.mean()
        assert 0.35 < avg_selected < 0.65  # Allow some variance

    def test_initialize_sparse_strategy(self, simple_evaluator, random_seed):
        """Test sparse initialization strategy."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=100, population_size=50, selector=selector)
        pop.initialize_population(strategy="sparse", sparsity=0.1)

        assert pop.chromosomes.shape == (50, 100)
        # Should have roughly 10% features selected
        avg_selected = pop.chromosomes.mean()
        assert avg_selected < 0.25  # Allow variance, but should be sparse

    def test_initialize_dense_strategy(self, simple_evaluator, random_seed):
        """Test dense initialization strategy."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=100, population_size=50, selector=selector)
        pop.initialize_population(strategy="dense", density=0.9)

        assert pop.chromosomes.shape == (50, 100)
        # Should have roughly 90% features selected
        avg_selected = pop.chromosomes.mean()
        assert avg_selected > 0.75  # Allow variance, but should be dense

    def test_initialize_unknown_strategy(self, simple_evaluator):
        """Test that unknown strategy raises ValueError."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=10, population_size=20, selector=selector)

        with pytest.raises(ValueError, match="Unknown strategy"):
            pop.initialize_population(strategy="invalid_strategy")

    def test_no_empty_chromosomes(self, simple_evaluator, random_seed):
        """Test that no chromosomes have zero features selected."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=10, population_size=100, selector=selector)

        # Try sparse to increase chance of empty chromosomes
        pop.initialize_population(strategy="sparse", sparsity=0.01)

        # All chromosomes must have at least one feature
        assert np.all(pop.chromosomes.sum(axis=1) >= 1)

    def test_measure_diversity_all_unique(self, simple_evaluator):
        """Test diversity measurement with all unique chromosomes."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=4, population_size=4, selector=selector)

        # Manually set all unique chromosomes
        pop.chromosomes = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        diversity = pop.measure_diversity()
        assert diversity == 100.0

    def test_measure_diversity_all_identical(self, simple_evaluator):
        """Test diversity measurement with all identical chromosomes."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=4, population_size=4, selector=selector)

        # All identical
        pop.chromosomes = np.array([
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
        ])

        diversity = pop.measure_diversity()
        assert diversity == 25.0  # 1 unique / 4 total * 100

    def test_measure_diversity_partial(self, simple_evaluator):
        """Test diversity measurement with partial duplicates."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=4, population_size=5, selector=selector)

        pop.chromosomes = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],  # Duplicate
            [0, 0, 1, 0],
            [0, 1, 0, 0],  # Duplicate
        ])

        diversity = pop.measure_diversity()
        assert diversity == 60.0  # 3 unique / 5 total * 100

    def test_add_single_chromosome(self, simple_evaluator):
        """Test adding a single chromosome."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=4, population_size=2, selector=selector)
        pop.initialize_population(strategy="random")

        initial_size = len(pop.chromosomes)
        new_chrom = np.array([1, 1, 0, 0])
        idx = pop.add_chromosome(new_chrom)

        assert len(pop.chromosomes) == initial_size + 1
        assert np.array_equal(pop.chromosomes[idx], new_chrom)

    def test_add_multiple_chromosomes(self, simple_evaluator):
        """Test adding multiple chromosomes at once."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=4, population_size=2, selector=selector)
        pop.initialize_population(strategy="random")

        initial_size = len(pop.chromosomes)
        new_chroms = np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ])
        idx = pop.add_chromosome(new_chroms)

        assert len(pop.chromosomes) == initial_size + 2
        assert np.array_equal(pop.chromosomes[-2:], new_chroms)

    def test_add_chromosome_to_empty_population(self, simple_evaluator):
        """Test adding chromosome to empty population."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=4, population_size=0, selector=selector)

        new_chrom = np.array([1, 0, 1, 0])
        idx = pop.add_chromosome(new_chrom)

        assert len(pop.chromosomes) == 1
        assert np.array_equal(pop.chromosomes[0], new_chrom)

    def test_select_parents(self, simple_evaluator, random_seed):
        """Test parent selection."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=4, population_size=10, selector=selector)
        pop.initialize_population(strategy="random")

        parent1, parent2 = pop.select_parents()

        assert parent1.shape == (4,)
        assert parent2.shape == (4,)
        assert np.all((parent1 == 0) | (parent1 == 1))
        assert np.all((parent2 == 0) | (parent2 == 1))

    def test_get_statistics(self, simple_evaluator):
        """Test population statistics."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=4, population_size=4, selector=selector)

        pop.chromosomes = np.array([
            [1, 0, 0, 0],  # 1 feature
            [1, 1, 0, 0],  # 2 features
            [1, 1, 1, 0],  # 3 features
            [1, 1, 1, 1],  # 4 features
        ])

        stats = pop.get_statistics()

        assert stats["size"] == 4
        assert stats["diversity"] == 100.0  # All unique
        assert stats["avg_features"] == 2.5
        assert stats["min_features"] == 1
        assert stats["max_features"] == 4

    def test_update_selector(self, simple_evaluator, random_seed):
        """Test that update_selector updates selector's population."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        pop = Population(num_features=4, population_size=5, selector=selector)
        pop.initialize_population(strategy="random")

        # Selector should have updated population
        assert selector.population is not None
        assert len(selector.population) == 5
        assert np.array_equal(selector.population, pop.chromosomes)

    def test_population_is_binary(self, simple_evaluator, random_seed):
        """Test that all generated chromosomes are strictly binary."""
        from fsga.selectors.tournament_selector import TournamentSelector

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        strategies = ["random", "uniform", "normal", "sparse", "dense"]

        for strategy in strategies:
            pop = Population(num_features=20, population_size=30, selector=selector)
            pop.initialize_population(strategy=strategy)

            # All values must be 0 or 1
            assert np.all((pop.chromosomes == 0) | (pop.chromosomes == 1)), \
                f"Strategy '{strategy}' produced non-binary values"
