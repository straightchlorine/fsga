"""Unit tests for crossover operators."""

import numpy as np
import pytest

from fsga.operators.uniform_crossover import UniformCrossover


class TestUniformCrossover:
    """Test suite for UniformCrossover."""

    def test_initialization(self):
        """Test crossover initialization."""
        crossover = UniformCrossover()
        assert crossover.dev is False

        crossover_dev = UniformCrossover(dev=True)
        assert crossover_dev.dev is True

    def test_crossover_returns_correct_shape(self, sample_chromosome):
        """Test that crossover returns two children with correct shape."""
        crossover = UniformCrossover()
        parent1 = sample_chromosome
        parent2 = np.array([0, 1, 0, 1])

        children = crossover.crossover(parent1, parent2)

        assert children.shape == (2, 4)

    def test_crossover_produces_binary_children(self, random_seed):
        """Test that children are binary (only 0s and 1s)."""
        crossover = UniformCrossover()
        parent1 = np.array([1, 1, 0, 0, 1])
        parent2 = np.array([0, 0, 1, 1, 0])

        children = crossover.crossover(parent1, parent2)

        assert np.all((children == 0) | (children == 1))

    def test_crossover_genes_from_parents(self, random_seed):
        """Test that each gene in children comes from one of the parents."""
        crossover = UniformCrossover()
        parent1 = np.array([1, 1, 0, 0, 1])
        parent2 = np.array([0, 0, 1, 1, 0])

        children = crossover.crossover(parent1, parent2)

        # Each gene in each child should match one of the parents at that position
        for child in children:
            for i in range(len(child)):
                assert child[i] in [parent1[i], parent2[i]]

    def test_crossover_complementary_children(self, random_seed):
        """Test that children are complementary (when parents differ at position i,
        child1[i] != child2[i])."""
        np.random.seed(42)
        crossover = UniformCrossover()
        parent1 = np.array([1, 1, 0, 0, 1])
        parent2 = np.array([0, 0, 1, 1, 0])

        children = crossover.crossover(parent1, parent2)
        child1, child2 = children[0], children[1]

        # Where parents differ, children should be complementary
        for i in range(len(parent1)):
            if parent1[i] != parent2[i]:
                assert child1[i] != child2[i]

    def test_crossover_preserves_chromosome_length(self):
        """Test crossover with various chromosome lengths."""
        crossover = UniformCrossover()

        for length in [1, 5, 10, 50, 100]:
            parent1 = np.random.randint(0, 2, size=length)
            parent2 = np.random.randint(0, 2, size=length)

            children = crossover.crossover(parent1, parent2)

            assert children.shape == (2, length)

    def test_crossover_with_identical_parents(self):
        """Test crossover when both parents are identical."""
        crossover = UniformCrossover()
        parent = np.array([1, 0, 1, 0, 1])

        children = crossover.crossover(parent, parent)

        # Both children should be identical to parents
        assert np.array_equal(children[0], parent)
        assert np.array_equal(children[1], parent)

    def test_crossover_randomness(self, random_seed):
        """Test that crossover produces different results across multiple runs."""
        crossover = UniformCrossover()
        parent1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        parent2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        results = []
        for _ in range(10):
            children = crossover.crossover(parent1, parent2)
            results.append(children[0].tolist())

        # Should get at least some variety
        unique_results = set(tuple(r) for r in results)
        assert len(unique_results) > 1

    def test_crossover_dev_mode(self, capsys):
        """Test debug output in dev mode."""
        crossover = UniformCrossover(dev=True)
        parent1 = np.array([1, 0])
        parent2 = np.array([0, 1])

        crossover.crossover(parent1, parent2)
        captured = capsys.readouterr()

        assert "Crossover on" in captured.out
        assert "Generated mask" in captured.out
        assert "Generated children" in captured.out

    def test_crossover_no_dev_output(self, capsys):
        """Test no debug output when dev=False."""
        crossover = UniformCrossover(dev=False)
        parent1 = np.array([1, 0])
        parent2 = np.array([0, 1])

        crossover.crossover(parent1, parent2)
        captured = capsys.readouterr()

        assert captured.out == ""

    def test_str_representation(self):
        """Test string representation."""
        crossover = UniformCrossover()
        assert str(crossover) == "UniformCrossover"

    def test_crossover_with_all_zeros(self):
        """Test crossover with all-zero parents."""
        crossover = UniformCrossover()
        parent1 = np.array([0, 0, 0, 0])
        parent2 = np.array([0, 0, 0, 0])

        children = crossover.crossover(parent1, parent2)

        assert np.all(children == 0)

    def test_crossover_with_all_ones(self):
        """Test crossover with all-one parents."""
        crossover = UniformCrossover()
        parent1 = np.array([1, 1, 1, 1])
        parent2 = np.array([1, 1, 1, 1])

        children = crossover.crossover(parent1, parent2)

        assert np.all(children == 1)

    def test_crossover_expected_mixing_ratio(self, random_seed):
        """Test that uniform crossover roughly mixes 50/50 from each parent."""
        crossover = UniformCrossover()
        parent1 = np.ones(100)
        parent2 = np.zeros(100)

        # Run multiple crossovers and check average
        total_from_parent1 = 0
        num_trials = 100

        for _ in range(num_trials):
            children = crossover.crossover(parent1, parent2)
            child1 = children[0]
            # Count how many 1s (from parent1) in child
            total_from_parent1 += child1.sum()

        # Average should be around 50% (50 genes from 100)
        avg_from_parent1 = total_from_parent1 / num_trials
        assert 40 < avg_from_parent1 < 60  # Allow some variance
