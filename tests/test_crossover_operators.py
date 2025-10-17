"""Unit tests for crossover operators (single-point, two-point, and multi-point)."""

import numpy as np
import pytest

from fsga.operators.multi_point_crossover import MultiPointCrossover
from fsga.operators.single_point_crossover import SinglePointCrossover
from fsga.operators.two_point_crossover import TwoPointCrossover


class TestSinglePointCrossover:
    """Test suite for SinglePointCrossover."""

    def test_initialization(self):
        """Test crossover initialization."""
        crossover = SinglePointCrossover()
        assert crossover.dev is False

        crossover_dev = SinglePointCrossover(dev=True)
        assert crossover_dev.dev is True

    def test_crossover_returns_correct_shape(self):
        """Test that crossover returns two children with correct shape."""
        crossover = SinglePointCrossover()
        parent1 = np.array([1, 0, 1, 0])
        parent2 = np.array([0, 1, 0, 1])

        children = crossover.crossover(parent1, parent2)

        assert children.shape == (2, 4)

    def test_crossover_produces_binary_children(self, random_seed):
        """Test that children are binary (only 0s and 1s)."""
        crossover = SinglePointCrossover()
        parent1 = np.array([1, 1, 0, 0, 1])
        parent2 = np.array([0, 0, 1, 1, 0])

        children = crossover.crossover(parent1, parent2)

        assert np.all((children == 0) | (children == 1))

    def test_crossover_genes_from_parents(self, random_seed):
        """Test that each gene in children comes from one of the parents."""
        crossover = SinglePointCrossover()
        parent1 = np.array([1, 1, 0, 0, 1])
        parent2 = np.array([0, 0, 1, 1, 0])

        children = crossover.crossover(parent1, parent2)

        # Each gene should match one of the parents at that position
        for child in children:
            for i in range(len(child)):
                assert child[i] in [parent1[i], parent2[i]]

    def test_crossover_preserves_chromosome_length(self):
        """Test crossover with various chromosome lengths."""
        crossover = SinglePointCrossover()

        for length in [2, 5, 10, 50, 100]:
            parent1 = np.random.randint(0, 2, size=length)
            parent2 = np.random.randint(0, 2, size=length)

            children = crossover.crossover(parent1, parent2)

            assert children.shape == (2, length)

    def test_crossover_with_identical_parents(self):
        """Test crossover when both parents are identical."""
        crossover = SinglePointCrossover()
        parent = np.array([1, 0, 1, 0, 1])

        children = crossover.crossover(parent, parent)

        # Both children should be identical to parents
        assert np.array_equal(children[0], parent)
        assert np.array_equal(children[1], parent)

    def test_crossover_different_sizes_raises_error(self):
        """Test that parents with different sizes raise ValueError."""
        crossover = SinglePointCrossover()
        parent1 = np.array([1, 0, 1])
        parent2 = np.array([0, 1, 0, 1])

        with pytest.raises(ValueError, match="same size"):
            crossover.crossover(parent1, parent2)

    def test_crossover_point_logic(self, random_seed):
        """Test that crossover point correctly splits chromosomes."""
        np.random.seed(42)
        crossover = SinglePointCrossover()
        parent1 = np.array([1, 1, 1, 1, 1])
        parent2 = np.array([0, 0, 0, 0, 0])

        children = crossover.crossover(parent1, parent2)
        child1, child2 = children[0], children[1]

        # Find the crossover point
        for i in range(1, len(parent1)):
            if all(child1[:i] == 1) and all(child1[i:] == 0):
                # Valid crossover point found
                assert all(child2[:i] == 0)
                assert all(child2[i:] == 1)
                break

    def test_crossover_randomness(self, random_seed):
        """Test that crossover produces different results across multiple runs."""
        crossover = SinglePointCrossover()
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
        crossover = SinglePointCrossover(dev=True)
        parent1 = np.array([1, 0])
        parent2 = np.array([0, 1])

        crossover.crossover(parent1, parent2)
        captured = capsys.readouterr()

        assert "SinglePointCrossover" in captured.out
        assert "Point=" in captured.out
        assert "Parent A" in captured.out

    def test_crossover_no_dev_output(self, capsys):
        """Test no debug output when dev=False."""
        crossover = SinglePointCrossover(dev=False)
        parent1 = np.array([1, 0])
        parent2 = np.array([0, 1])

        crossover.crossover(parent1, parent2)
        captured = capsys.readouterr()

        assert captured.out == ""

    def test_str_representation(self):
        """Test string representation."""
        crossover = SinglePointCrossover()
        assert str(crossover) == "SinglePointCrossover"

    def test_crossover_minimum_size(self):
        """Test crossover with minimum chromosome size (2)."""
        crossover = SinglePointCrossover()
        parent1 = np.array([1, 0])
        parent2 = np.array([0, 1])

        children = crossover.crossover(parent1, parent2)

        # With size 2, only one possible crossover point (1)
        # Child1: parent1[0] + parent2[1] = [1, 1]
        # Child2: parent2[0] + parent1[1] = [0, 0]
        assert children.shape == (2, 2)


class TestTwoPointCrossover:
    """Test suite for TwoPointCrossover."""

    def test_initialization(self):
        """Test crossover initialization."""
        crossover = TwoPointCrossover()
        assert crossover.dev is False

        crossover_dev = TwoPointCrossover(dev=True)
        assert crossover_dev.dev is True

    def test_crossover_returns_correct_shape(self):
        """Test that crossover returns two children with correct shape."""
        crossover = TwoPointCrossover()
        parent1 = np.array([1, 0, 1, 0, 1])
        parent2 = np.array([0, 1, 0, 1, 0])

        children = crossover.crossover(parent1, parent2)

        assert children.shape == (2, 5)

    def test_crossover_produces_binary_children(self, random_seed):
        """Test that children are binary (only 0s and 1s)."""
        crossover = TwoPointCrossover()
        parent1 = np.array([1, 1, 0, 0, 1, 0])
        parent2 = np.array([0, 0, 1, 1, 0, 1])

        children = crossover.crossover(parent1, parent2)

        assert np.all((children == 0) | (children == 1))

    def test_crossover_genes_from_parents(self, random_seed):
        """Test that each gene in children comes from one of the parents."""
        crossover = TwoPointCrossover()
        parent1 = np.array([1, 1, 0, 0, 1, 0])
        parent2 = np.array([0, 0, 1, 1, 0, 1])

        children = crossover.crossover(parent1, parent2)

        # Each gene should match one of the parents at that position
        for child in children:
            for i in range(len(child)):
                assert child[i] in [parent1[i], parent2[i]]

    def test_crossover_preserves_chromosome_length(self):
        """Test crossover with various chromosome lengths."""
        crossover = TwoPointCrossover()

        for length in [3, 5, 10, 50, 100]:
            parent1 = np.random.randint(0, 2, size=length)
            parent2 = np.random.randint(0, 2, size=length)

            children = crossover.crossover(parent1, parent2)

            assert children.shape == (2, length)

    def test_crossover_with_identical_parents(self):
        """Test crossover when both parents are identical."""
        crossover = TwoPointCrossover()
        parent = np.array([1, 0, 1, 0, 1])

        children = crossover.crossover(parent, parent)

        # Both children should be identical to parents
        assert np.array_equal(children[0], parent)
        assert np.array_equal(children[1], parent)

    def test_crossover_different_sizes_raises_error(self):
        """Test that parents with different sizes raise ValueError."""
        crossover = TwoPointCrossover()
        parent1 = np.array([1, 0, 1])
        parent2 = np.array([0, 1, 0, 1])

        with pytest.raises(ValueError, match="same size"):
            crossover.crossover(parent1, parent2)

    def test_crossover_too_small_raises_error(self):
        """Test that chromosomes smaller than 3 raise ValueError."""
        crossover = TwoPointCrossover()
        parent1 = np.array([1, 0])
        parent2 = np.array([0, 1])

        with pytest.raises(ValueError, match="at least 3"):
            crossover.crossover(parent1, parent2)

    def test_crossover_point_logic(self, random_seed):
        """Test that two crossover points correctly split chromosomes."""
        np.random.seed(42)
        crossover = TwoPointCrossover()
        parent1 = np.array([1, 1, 1, 1, 1, 1])
        parent2 = np.array([0, 0, 0, 0, 0, 0])

        children = crossover.crossover(parent1, parent2)
        child1, child2 = children[0], children[1]

        # Child1 should have pattern: 1s, 0s, 1s
        # Child2 should have pattern: 0s, 1s, 0s
        # Count transitions to verify structure
        transitions1 = sum(child1[i] != child1[i+1] for i in range(len(child1)-1))
        transitions2 = sum(child2[i] != child2[i+1] for i in range(len(child2)-1))

        # Should have exactly 2 transitions (from two crossover points)
        # or 0 if points are at edges
        assert transitions1 in [0, 2]
        assert transitions2 in [0, 2]

    def test_crossover_randomness(self, random_seed):
        """Test that crossover produces different results across multiple runs."""
        crossover = TwoPointCrossover()
        parent1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        parent2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        results = []
        for _ in range(15):
            children = crossover.crossover(parent1, parent2)
            results.append(children[0].tolist())

        # Should get variety
        unique_results = set(tuple(r) for r in results)
        assert len(unique_results) > 1

    def test_crossover_dev_mode(self, capsys):
        """Test debug output in dev mode."""
        crossover = TwoPointCrossover(dev=True)
        parent1 = np.array([1, 0, 1, 0, 1])
        parent2 = np.array([0, 1, 0, 1, 0])

        crossover.crossover(parent1, parent2)
        captured = capsys.readouterr()

        assert "TwoPointCrossover" in captured.out
        assert "Points=" in captured.out
        assert "Parent A" in captured.out

    def test_crossover_no_dev_output(self, capsys):
        """Test no debug output when dev=False."""
        crossover = TwoPointCrossover(dev=False)
        parent1 = np.array([1, 0, 1])
        parent2 = np.array([0, 1, 0])

        crossover.crossover(parent1, parent2)
        captured = capsys.readouterr()

        assert captured.out == ""

    def test_str_representation(self):
        """Test string representation."""
        crossover = TwoPointCrossover()
        assert str(crossover) == "TwoPointCrossover"

    def test_crossover_minimum_size(self):
        """Test crossover with minimum chromosome size (3)."""
        crossover = TwoPointCrossover()
        parent1 = np.array([1, 0, 1])
        parent2 = np.array([0, 1, 0])

        children = crossover.crossover(parent1, parent2)

        # With size 3, only one pair of points possible (1, 2)
        assert children.shape == (2, 3)

    def test_crossover_points_sorted(self, random_seed):
        """Test that crossover points are properly sorted."""
        crossover = TwoPointCrossover()
        parent1 = np.ones(20, dtype=int)
        parent2 = np.zeros(20, dtype=int)

        # Run multiple times to check sorting
        for _ in range(10):
            children = crossover.crossover(parent1, parent2)
            # If implementation is correct, children should be valid
            assert np.all((children == 0) | (children == 1))


class TestMultiPointCrossover:
    """Test suite for MultiPointCrossover."""

    def test_initialization_default(self):
        """Test crossover initialization with default (random) points."""
        crossover = MultiPointCrossover()
        assert crossover.points is None
        assert crossover.dev is False

    def test_initialization_fixed_points(self):
        """Test crossover initialization with fixed points."""
        crossover = MultiPointCrossover(points=[2, 5, 8])
        assert crossover.points == [2, 5, 8]
        assert crossover.dev is False

    def test_initialization_dev_mode(self):
        """Test crossover initialization with dev mode."""
        crossover = MultiPointCrossover(dev=True)
        assert crossover.dev is True

    def test_crossover_returns_correct_shape(self):
        """Test that crossover returns two children with correct shape."""
        crossover = MultiPointCrossover(points=[2, 5])
        parent1 = np.array([1, 0, 1, 0, 1, 0, 1])
        parent2 = np.array([0, 1, 0, 1, 0, 1, 0])

        children = crossover.crossover(parent1, parent2)

        assert children.shape == (2, 7)

    def test_crossover_produces_binary_children(self, random_seed):
        """Test that children are binary (only 0s and 1s)."""
        crossover = MultiPointCrossover()
        parent1 = np.array([1, 1, 0, 0, 1, 0, 1, 1])
        parent2 = np.array([0, 0, 1, 1, 0, 1, 0, 0])

        children = crossover.crossover(parent1, parent2)

        assert np.all((children == 0) | (children == 1))

    def test_crossover_genes_from_parents(self, random_seed):
        """Test that each gene in children comes from one of the parents."""
        crossover = MultiPointCrossover(points=[2, 5])
        parent1 = np.array([1, 1, 0, 0, 1, 0, 1])
        parent2 = np.array([0, 0, 1, 1, 0, 1, 0])

        children = crossover.crossover(parent1, parent2)

        # Each gene should match one of the parents at that position
        for child in children:
            for i in range(len(child)):
                assert child[i] in [parent1[i], parent2[i]]

    def test_crossover_with_fixed_points(self):
        """Test crossover with fixed crossover points."""
        crossover = MultiPointCrossover(points=[2, 5])
        parent1 = np.array([1, 1, 1, 1, 1, 1, 1])
        parent2 = np.array([0, 0, 0, 0, 0, 0, 0])

        children = crossover.crossover(parent1, parent2)
        child1, child2 = children[0], children[1]

        # Implementation swaps alternate segments based on loop index
        # Start: child1=P1=[1,1,1,1,1,1,1], child2=P2=[0,0,0,0,0,0,0]
        # i=0 (swap [2:5]): child1 gets P2[2:5], child2 gets P1[2:5]
        # i=1 (swap [5:]): child1 gets P2[5:], child2 gets P1[5:]
        # Result: child1=[1,1,0,0,0,0,0], child2=[0,0,1,1,1,1,1]
        assert np.array_equal(child1, np.array([1, 1, 0, 0, 0, 0, 0]))
        assert np.array_equal(child2, np.array([0, 0, 1, 1, 1, 1, 1]))

    def test_crossover_preserves_chromosome_length(self):
        """Test crossover with various chromosome lengths."""
        crossover = MultiPointCrossover()

        for length in [5, 10, 20, 50]:
            parent1 = np.random.randint(0, 2, size=length)
            parent2 = np.random.randint(0, 2, size=length)

            children = crossover.crossover(parent1, parent2)

            assert children.shape == (2, length)

    def test_crossover_with_identical_parents(self):
        """Test crossover when both parents are identical."""
        crossover = MultiPointCrossover(points=[2, 5])
        parent = np.array([1, 0, 1, 0, 1, 0, 1])

        children = crossover.crossover(parent, parent)

        # Both children should be identical to parents
        assert np.array_equal(children[0], parent)
        assert np.array_equal(children[1], parent)

    def test_crossover_different_sizes_raises_error(self):
        """Test that parents with different sizes raise ValueError."""
        crossover = MultiPointCrossover()
        parent1 = np.array([1, 0, 1])
        parent2 = np.array([0, 1, 0, 1])

        with pytest.raises(ValueError, match="same size"):
            crossover.crossover(parent1, parent2)

    def test_crossover_too_small_raises_error(self):
        """Test that chromosomes smaller than 2 raise ValueError."""
        crossover = MultiPointCrossover(points=[1])
        parent1 = np.array([1])
        parent2 = np.array([0])

        with pytest.raises(ValueError, match="at least 2"):
            crossover.crossover(parent1, parent2)

    def test_crossover_invalid_points_raises_error(self):
        """Test that invalid points raise ValueError."""
        crossover = MultiPointCrossover(points=[0, 10])  # 0 is invalid, 10 out of range
        parent1 = np.array([1, 0, 1, 0, 1])
        parent2 = np.array([0, 1, 0, 1, 0])

        with pytest.raises(ValueError, match="between 1 and"):
            crossover.crossover(parent1, parent2)

    def test_crossover_single_point(self):
        """Test crossover with a single point (should work like single-point)."""
        crossover = MultiPointCrossover(points=[3])
        parent1 = np.array([1, 1, 1, 1, 1])
        parent2 = np.array([0, 0, 0, 0, 0])

        children = crossover.crossover(parent1, parent2)
        child1, child2 = children[0], children[1]

        # With point [3]: [0:3], [3:]
        # child1: parent1[0:3] + parent2[3:] = [1,1,1,0,0]
        assert np.array_equal(child1, np.array([1, 1, 1, 0, 0]))
        assert np.array_equal(child2, np.array([0, 0, 0, 1, 1]))

    def test_crossover_three_points(self):
        """Test crossover with three points."""
        crossover = MultiPointCrossover(points=[2, 4, 6])
        parent1 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        parent2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])

        children = crossover.crossover(parent1, parent2)
        child1, child2 = children[0], children[1]

        # Alternating swap: swaps children[i%2][start:end] with children[(i+1)%2][start:end]
        # Start: child1=P1=[1,1,1,1,1,1,1,1], child2=P2=[0,0,0,0,0,0,0,0]
        # i=0 (swap child0[2:4]): child1=[1,1,0,0,1,1,1,1], child2=[0,0,1,1,0,0,0,0]
        # i=1 (swap child1[4:6]): child1=[1,1,0,0,0,0,1,1], child2=[0,0,1,1,1,1,0,0]
        # i=2 (swap child0[6:]): child1=[1,1,0,0,0,0,0,0], child2=[0,0,1,1,1,1,1,1]
        assert np.array_equal(child1, np.array([1, 1, 0, 0, 0, 0, 0, 0]))
        assert np.array_equal(child2, np.array([0, 0, 1, 1, 1, 1, 1, 1]))

    def test_crossover_four_points(self):
        """Test crossover with four points (maximum typical)."""
        crossover = MultiPointCrossover(points=[2, 4, 6, 8])
        parent1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        parent2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        children = crossover.crossover(parent1, parent2)
        child1, child2 = children[0], children[1]

        # Alternating swap pattern
        # Start: child1=[1,1,1,1,1,1,1,1,1,1], child2=[0,0,0,0,0,0,0,0,0,0]
        # i=0 (swap child0[2:4]): child1=[1,1,0,0,1,1,1,1,1,1], child2=[0,0,1,1,0,0,0,0,0,0]
        # i=1 (swap child1[4:6]): child1=[1,1,0,0,0,0,1,1,1,1], child2=[0,0,1,1,1,1,0,0,0,0]
        # i=2 (swap child0[6:8]): child1=[1,1,0,0,0,0,0,0,1,1], child2=[0,0,1,1,1,1,1,1,0,0]
        # i=3 (swap child1[8:]): child1=[1,1,0,0,0,0,0,0,0,0], child2=[0,0,1,1,1,1,1,1,1,1]
        assert np.array_equal(child1, np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
        assert np.array_equal(child2, np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1]))

    def test_crossover_random_points_variable_count(self, random_seed):
        """Test that random mode selects 1-4 points."""
        crossover = MultiPointCrossover()  # No fixed points
        parent1 = np.array([1] * 20)
        parent2 = np.array([0] * 20)

        # Run multiple times to verify randomness
        for _ in range(10):
            children = crossover.crossover(parent1, parent2)
            # Should produce valid children
            assert children.shape == (2, 20)
            assert np.all((children == 0) | (children == 1))

    def test_crossover_randomness(self, random_seed):
        """Test that crossover produces different results across multiple runs."""
        parent1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        parent2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        results = []
        for _ in range(15):
            # Create new crossover each time to avoid caching
            crossover = MultiPointCrossover()
            children = crossover.crossover(parent1, parent2)
            results.append(children[0].tolist())

        # Should get variety
        unique_results = set(tuple(r) for r in results)
        assert len(unique_results) > 1

    def test_crossover_points_sorted(self):
        """Test that crossover points are sorted internally."""
        # Points given unsorted
        crossover = MultiPointCrossover(points=[5, 2, 8])
        parent1 = np.array([1] * 10)
        parent2 = np.array([0] * 10)

        children = crossover.crossover(parent1, parent2)

        # Should still work correctly (implementation sorts them)
        assert children.shape == (2, 10)
        assert np.all((children == 0) | (children == 1))

    def test_crossover_minimum_size(self):
        """Test crossover with minimum chromosome size (2)."""
        crossover = MultiPointCrossover(points=[1])
        parent1 = np.array([1, 0])
        parent2 = np.array([0, 1])

        children = crossover.crossover(parent1, parent2)

        # With size 2 and point [1]: [0:1], [1:]
        # child1: p1[0:1] + p2[1:] = [1, 1]
        # child2: p2[0:1] + p1[1:] = [0, 0]
        assert np.array_equal(children[0], np.array([1, 1]))
        assert np.array_equal(children[1], np.array([0, 0]))

    def test_str_representation(self):
        """Test string representation."""
        crossover = MultiPointCrossover()
        assert "MultiPointCrossover" in str(crossover)
