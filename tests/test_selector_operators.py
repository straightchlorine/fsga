"""Unit tests for selector operators (roulette, ranking, elitism)."""

import numpy as np
import pytest

from fsga.selectors.roulette_selector import RouletteSelector
from fsga.selectors.ranking_selector import RankingSelector
from fsga.selectors.elitism_selector import ElitismSelector


class TestRouletteSelector:
    """Test suite for RouletteSelector."""

    def test_initialization(self, simple_evaluator):
        """Test selector initialization."""
        selector = RouletteSelector(simple_evaluator)

        assert selector.evaluator is simple_evaluator
        assert selector.number_of_parents == 2
        assert selector._population is None

    def test_select_returns_two_parents(self, simple_evaluator, sample_population):
        """Test that select returns two parent chromosomes."""
        selector = RouletteSelector(simple_evaluator)
        selector.population = sample_population

        parent1, parent2 = selector.select()

        assert parent1.shape == (4,)
        assert parent2.shape == (4,)

    def test_select_parents_from_population(self, simple_evaluator, sample_population):
        """Test that selected parents come from the population."""
        selector = RouletteSelector(simple_evaluator)
        selector.population = sample_population

        parent1, parent2 = selector.select()

        # Check parents are in population
        found1 = any(np.array_equal(parent1, chrom) for chrom in sample_population)
        found2 = any(np.array_equal(parent2, chrom) for chrom in sample_population)
        assert found1
        assert found2

    def test_select_favors_higher_fitness(self, simple_evaluator, random_seed):
        """Test that roulette selection favors higher fitness individuals."""
        # Create population with clear fitness gradient
        population = np.array([
            [0, 0, 0, 0],  # fitness = 0
            [1, 0, 0, 0],  # fitness = 1
            [1, 1, 0, 0],  # fitness = 2
            [1, 1, 1, 0],  # fitness = 3
            [1, 1, 1, 1],  # fitness = 4 (best)
        ])

        selector = RouletteSelector(simple_evaluator)
        selector.population = population

        # Run many selections and count how often best individual is selected
        best_count = 0
        num_trials = 200

        for _ in range(num_trials):
            parent1, parent2 = selector.select()
            if np.array_equal(parent1, population[-1]):
                best_count += 1
            if np.array_equal(parent2, population[-1]):
                best_count += 1

        # Best individual (fitness=4) should be selected more often
        # Theoretical probability: 4/(0+1+2+3+4) = 4/10 = 40%
        selection_rate = best_count / (num_trials * 2)
        assert selection_rate > 0.25  # Should be significantly selected

    def test_select_handles_zero_fitness(self, simple_evaluator):
        """Test handling of zero/negative fitness."""
        # All zeros - no features selected
        population = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])

        selector = RouletteSelector(simple_evaluator)
        selector.population = population

        # Should not crash (shifts fitness values)
        parent1, parent2 = selector.select()
        assert parent1 is not None
        assert parent2 is not None

    def test_str_representation(self, simple_evaluator):
        """Test string representation."""
        selector = RouletteSelector(simple_evaluator)
        assert "RouletteSelector" in str(selector)

    def test_select_requires_population(self, simple_evaluator):
        """Test that select fails without population."""
        selector = RouletteSelector(simple_evaluator)

        with pytest.raises(ValueError, match="Population is not set"):
            selector.select()

    def test_select_deterministic_with_seed(self, simple_evaluator, sample_population):
        """Test that selection is deterministic with same random seed."""
        selector = RouletteSelector(simple_evaluator)
        selector.population = sample_population

        np.random.seed(42)
        parent1_a, parent2_a = selector.select()

        np.random.seed(42)
        parent1_b, parent2_b = selector.select()

        assert np.array_equal(parent1_a, parent1_b)
        assert np.array_equal(parent2_a, parent2_b)


class TestRankingSelector:
    """Test suite for RankingSelector."""

    def test_initialization(self, simple_evaluator):
        """Test selector initialization."""
        selector = RankingSelector(simple_evaluator, scale_factor=2.0)

        assert selector.evaluator is simple_evaluator
        assert selector.scale_factor == 2.0
        assert selector.number_of_parents == 2

    def test_select_returns_two_parents(self, simple_evaluator, sample_population):
        """Test that select returns two parent chromosomes."""
        selector = RankingSelector(simple_evaluator)
        selector.population = sample_population

        parent1, parent2 = selector.select()

        assert parent1.shape == (4,)
        assert parent2.shape == (4,)

    def test_select_parents_from_population(self, simple_evaluator, sample_population):
        """Test that selected parents come from the population."""
        selector = RankingSelector(simple_evaluator)
        selector.population = sample_population

        parent1, parent2 = selector.select()

        found1 = any(np.array_equal(parent1, chrom) for chrom in sample_population)
        found2 = any(np.array_equal(parent2, chrom) for chrom in sample_population)
        assert found1
        assert found2

    def test_select_favors_higher_fitness(self, simple_evaluator, random_seed):
        """Test that ranking selection favors higher fitness individuals."""
        population = np.array([
            [0, 0, 0, 0],  # fitness = 0
            [1, 0, 0, 0],  # fitness = 1
            [1, 1, 0, 0],  # fitness = 2
            [1, 1, 1, 0],  # fitness = 3
            [1, 1, 1, 1],  # fitness = 4 (best)
        ])

        selector = RankingSelector(simple_evaluator, scale_factor=1.0)
        selector.population = population

        best_count = 0
        num_trials = 200

        for _ in range(num_trials):
            parent1, parent2 = selector.select()
            if np.array_equal(parent1, population[-1]):
                best_count += 1
            if np.array_equal(parent2, population[-1]):
                best_count += 1

        selection_rate = best_count / (num_trials * 2)
        assert selection_rate > 0.15  # Should have higher probability

    def test_scale_factor_effect(self, simple_evaluator, random_seed):
        """Test that scale_factor affects selection pressure."""
        population = np.array([
            [1, 1, 1, 1],  # Best
            [0, 0, 0, 0],  # Worst
        ])

        # Higher scale_factor = lower selection pressure (more uniform)
        selector_high = RankingSelector(simple_evaluator, scale_factor=10.0)
        selector_high.population = population

        # Lower scale_factor = higher selection pressure
        selector_low = RankingSelector(simple_evaluator, scale_factor=0.5)
        selector_low.population = population

        # Both should work
        p1, p2 = selector_high.select()
        assert p1 is not None

        p1, p2 = selector_low.select()
        assert p1 is not None

    def test_str_representation(self, simple_evaluator):
        """Test string representation."""
        selector = RankingSelector(simple_evaluator, scale_factor=2.5)
        assert "RankingSelector" in str(selector)
        assert "2.5" in str(selector)

    def test_select_requires_population(self, simple_evaluator):
        """Test that select fails without population."""
        selector = RankingSelector(simple_evaluator)

        with pytest.raises(ValueError, match="Population is not set"):
            selector.select()


class TestElitismSelector:
    """Test suite for ElitismSelector."""

    def test_initialization(self, simple_evaluator):
        """Test selector initialization."""
        selector = ElitismSelector(simple_evaluator, n_elite=3)

        assert selector.evaluator is simple_evaluator
        assert selector.n_elite == 3

    def test_select_returns_two_parents(self, simple_evaluator, sample_population):
        """Test that select returns two parent chromosomes."""
        selector = ElitismSelector(simple_evaluator)
        selector.population = sample_population

        parent1, parent2 = selector.select()

        assert parent1.shape == (4,)
        assert parent2.shape == (4,)

    def test_select_returns_best_individuals(self, simple_evaluator):
        """Test that elitism selects the best individuals."""
        # Create population with clear fitness ordering
        population = np.array([
            [0, 0, 0, 0],  # fitness = 0 (worst)
            [1, 0, 0, 0],  # fitness = 1
            [1, 1, 0, 0],  # fitness = 2
            [1, 1, 1, 0],  # fitness = 3 (2nd best)
            [1, 1, 1, 1],  # fitness = 4 (best)
        ])

        selector = ElitismSelector(simple_evaluator, n_elite=2)
        selector.population = population

        parent1, parent2 = selector.select()

        # Should select the two best
        assert np.array_equal(parent1, population[-1])  # Best
        assert np.array_equal(parent2, population[-2])  # 2nd best

    def test_select_deterministic(self, simple_evaluator, sample_population):
        """Test that elitism selection is deterministic."""
        selector = ElitismSelector(simple_evaluator)
        selector.population = sample_population

        parent1_a, parent2_a = selector.select()
        parent1_b, parent2_b = selector.select()

        # Should always return same parents
        assert np.array_equal(parent1_a, parent1_b)
        assert np.array_equal(parent2_a, parent2_b)

    def test_select_with_small_population(self, simple_evaluator):
        """Test elitism with population smaller than n_elite."""
        population = np.array([[1, 0]])  # Only 1 individual

        selector = ElitismSelector(simple_evaluator, n_elite=2)
        selector.population = population

        parent1, parent2 = selector.select()

        # Should duplicate the only individual
        assert np.array_equal(parent1, population[0])
        assert np.array_equal(parent2, population[0])

    def test_str_representation(self, simple_evaluator):
        """Test string representation."""
        selector = ElitismSelector(simple_evaluator, n_elite=5)
        assert "ElitismSelector" in str(selector)
        assert "5" in str(selector)

    def test_select_requires_population(self, simple_evaluator):
        """Test that select fails without population."""
        selector = ElitismSelector(simple_evaluator)

        with pytest.raises(ValueError, match="Population is not set"):
            selector.select()

    def test_select_always_best(self, simple_evaluator, random_seed):
        """Test that elitism always selects best, regardless of randomness."""
        population = np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],  # Best
            [1, 1, 0, 0],
        ])

        selector = ElitismSelector(simple_evaluator)
        selector.population = population

        # Run multiple times
        for _ in range(10):
            parent1, parent2 = selector.select()
            # First parent should always be the best
            assert np.array_equal(parent1, population[1])
