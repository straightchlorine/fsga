"""Unit tests for selector operators."""

import numpy as np
import pytest

from fsga.selectors.tournament_selector import TournamentSelector


class TestTournamentSelector:
    """Test suite for TournamentSelector."""

    def test_initialization(self, simple_evaluator):
        """Test selector initialization."""
        selector = TournamentSelector(simple_evaluator, tournament_size=3)

        assert selector.evaluator is simple_evaluator
        assert selector.tournament_size == 3
        assert selector._population is None

    def test_tournament_size_validation(self, simple_evaluator):
        """Test that tournament size can be set to any value (validated at select time)."""
        # TournamentSelector allows any tournament_size at init, validates during select
        selector = TournamentSelector(simple_evaluator, tournament_size=1)
        assert selector.tournament_size == 1

    def test_tournament_size_setter(self, simple_evaluator):
        """Test tournament size can be updated."""
        selector = TournamentSelector(simple_evaluator, tournament_size=3)

        selector.tournament_size = 5
        assert selector.tournament_size == 5

        # Tournament size can be set to any value (validated during select)
        selector.tournament_size = 1
        assert selector.tournament_size == 1

    def test_select_returns_two_parents(self, simple_evaluator, sample_population):
        """Test that select returns two parent chromosomes."""
        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        selector.population = sample_population

        parent1, parent2 = selector.select()

        assert parent1.shape == (4,)
        assert parent2.shape == (4,)
        assert np.all((parent1 == 0) | (parent1 == 1))
        assert np.all((parent2 == 0) | (parent2 == 1))

    def test_select_parents_from_population(self, simple_evaluator, sample_population):
        """Test that selected parents come from the population."""
        selector = TournamentSelector(simple_evaluator, tournament_size=2)
        selector.population = sample_population

        parent1, parent2 = selector.select()

        # Check parent1 is in population
        found1 = any(np.array_equal(parent1, chrom) for chrom in sample_population)
        assert found1

        # Check parent2 is in population
        found2 = any(np.array_equal(parent2, chrom) for chrom in sample_population)
        assert found2

    def test_select_requires_population(self, simple_evaluator):
        """Test that select fails without population."""
        selector = TournamentSelector(simple_evaluator, tournament_size=3)

        with pytest.raises(ValueError, match="Population is not set"):
            selector.select()

    def test_tournament_size_cannot_exceed_population(self, simple_evaluator):
        """Test that tournament size cannot exceed population size."""
        selector = TournamentSelector(simple_evaluator, tournament_size=10)
        small_population = np.array([[1, 0], [0, 1], [1, 1]])  # Only 3 individuals

        selector.population = small_population

        with pytest.raises(ValueError, match="Tournament size.*cannot be larger"):
            selector.select()

    def test_select_favors_higher_fitness(self, simple_evaluator, random_seed):
        """Test that tournament selection tends to select higher fitness individuals."""
        # Create population with clear fitness gradient
        population = np.array([
            [0, 0, 0, 0],  # fitness = 0
            [1, 0, 0, 0],  # fitness = 1
            [1, 1, 0, 0],  # fitness = 2
            [1, 1, 1, 0],  # fitness = 3
            [1, 1, 1, 1],  # fitness = 4 (best)
        ])

        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        selector.population = population

        # Run many selections and count how often best individual is selected
        best_count = 0
        num_trials = 100

        for _ in range(num_trials):
            parent1, parent2 = selector.select()
            if np.array_equal(parent1, population[-1]):
                best_count += 1
            if np.array_equal(parent2, population[-1]):
                best_count += 1

        # With tournament size 3 and clear fitness gradient,
        # best individual should be selected more often than random (40%)
        selection_rate = best_count / (num_trials * 2)
        assert selection_rate > 0.4

    def test_select_with_minimum_tournament_size(self, simple_evaluator, sample_population):
        """Test selection with minimum tournament size (2)."""
        selector = TournamentSelector(simple_evaluator, tournament_size=2)
        selector.population = sample_population

        parent1, parent2 = selector.select()

        assert parent1.shape == (4,)
        assert parent2.shape == (4,)

    def test_select_with_large_tournament(self, simple_evaluator):
        """Test selection with large tournament size."""
        population = np.random.randint(0, 2, size=(50, 10))
        selector = TournamentSelector(simple_evaluator, tournament_size=10)
        selector.population = population

        parent1, parent2 = selector.select()

        assert parent1.shape == (10,)
        assert parent2.shape == (10,)

    def test_population_setter(self, simple_evaluator, sample_population):
        """Test population property setter."""
        selector = TournamentSelector(simple_evaluator, tournament_size=3)

        selector.population = sample_population

        assert selector.population is not None
        assert np.array_equal(selector.population, sample_population)

    def test_str_representation(self, simple_evaluator):
        """Test string representation."""
        selector = TournamentSelector(simple_evaluator, tournament_size=5)

        assert "TournamentSelector" in str(selector)

    def test_select_different_parents(self, simple_evaluator, random_seed):
        """Test that two different parents can be selected."""
        # Create diverse population
        population = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
        ])

        selector = TournamentSelector(simple_evaluator, tournament_size=2)
        selector.population = population

        # Run multiple times, should get different parents at least sometimes
        different_parents = False
        for _ in range(20):
            parent1, parent2 = selector.select()
            if not np.array_equal(parent1, parent2):
                different_parents = True
                break

        assert different_parents

    def test_select_deterministic_with_seed(self, simple_evaluator, sample_population):
        """Test that selection is deterministic with same random seed."""
        selector = TournamentSelector(simple_evaluator, tournament_size=3)
        selector.population = sample_population

        np.random.seed(42)
        parent1_a, parent2_a = selector.select()

        np.random.seed(42)
        parent1_b, parent2_b = selector.select()

        assert np.array_equal(parent1_a, parent1_b)
        assert np.array_equal(parent2_a, parent2_b)

    def test_evaluator_property(self, simple_evaluator):
        """Test evaluator property getter and setter."""
        selector = TournamentSelector(simple_evaluator, tournament_size=3)

        assert selector.evaluator is simple_evaluator

        # Create new evaluator
        class NewEvaluator:
            def evaluate(self, chromosome):
                return 1.0

        new_eval = NewEvaluator()
        selector.evaluator = new_eval

        assert selector.evaluator is new_eval
