"""Unit tests for mutation operators."""

import numpy as np
import pytest

from fsga.mutations.bitflip_mutation import BitFlipMutation


class TestBitFlipMutation:
    """Test suite for BitFlipMutation."""

    def test_initialization(self):
        """Test mutation initialization."""
        mutation = BitFlipMutation(probability=0.1)
        assert mutation.probability == 0.1

    def test_mutate_returns_correct_shape(self, sample_population):
        """Test that mutation returns same shape as input."""
        mutation = BitFlipMutation(probability=0.1)

        mutated = mutation.mutate(sample_population)

        assert mutated.shape == sample_population.shape

    def test_mutate_does_not_modify_input(self, sample_population):
        """Test that mutation does not modify the input population."""
        mutation = BitFlipMutation(probability=0.5)
        original = sample_population.copy()

        mutation.mutate(sample_population)

        assert np.array_equal(sample_population, original)

    def test_mutate_produces_binary_output(self, sample_population):
        """Test that mutated chromosomes are still binary."""
        mutation = BitFlipMutation(probability=0.5)

        mutated = mutation.mutate(sample_population)

        assert np.all((mutated == 0) | (mutated == 1))

    def test_mutate_with_zero_probability(self, sample_population):
        """Test mutation with 0% probability (no changes)."""
        mutation = BitFlipMutation(probability=0.0)

        mutated = mutation.mutate(sample_population)

        assert np.array_equal(mutated, sample_population)

    def test_mutate_with_full_probability(self, random_seed):
        """Test mutation with 100% probability (all bits flip)."""
        mutation = BitFlipMutation(probability=1.0)
        population = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])

        mutated = mutation.mutate(population)

        # All bits should be flipped
        expected = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
        assert np.array_equal(mutated, expected)

    def test_mutate_expected_flip_rate(self, random_seed):
        """Test that mutation rate matches expected flip rate."""
        prob = 0.1
        mutation = BitFlipMutation(probability=prob)

        # Large population to get good statistics
        population = np.ones((100, 100), dtype=int)

        mutated = mutation.mutate(population)

        # Count flipped bits (should be ~10%)
        flipped = (population != mutated).sum()
        total_bits = population.size
        flip_rate = flipped / total_bits

        # Allow 20% variance (0.08 to 0.12 for prob=0.1)
        assert prob * 0.8 < flip_rate < prob * 1.2

    def test_mutate_single_chromosome(self):
        """Test mutation on single chromosome (1D array)."""
        mutation = BitFlipMutation(probability=0.5)
        chromosome = np.array([[1, 0, 1, 0]])  # 2D with 1 row

        mutated = mutation.mutate(chromosome)

        assert mutated.shape == (1, 4)
        assert np.all((mutated == 0) | (mutated == 1))

    def test_mutate_various_population_sizes(self, random_seed):
        """Test mutation with various population and chromosome sizes."""
        mutation = BitFlipMutation(probability=0.1)

        sizes = [(1, 5), (10, 10), (50, 20), (5, 100)]

        for pop_size, chrom_len in sizes:
            population = np.random.randint(0, 2, size=(pop_size, chrom_len))
            mutated = mutation.mutate(population)

            assert mutated.shape == (pop_size, chrom_len)
            assert np.all((mutated == 0) | (mutated == 1))

    def test_mutate_generation_parameter_unused(self, sample_population):
        """Test that generation parameter doesn't affect BitFlipMutation."""
        mutation = BitFlipMutation(probability=0.1)
        np.random.seed(42)
        mutated1 = mutation.mutate(sample_population, generation=0)

        np.random.seed(42)
        mutated2 = mutation.mutate(sample_population, generation=100)

        # Results should be identical (generation ignored for basic bit-flip)
        assert np.array_equal(mutated1, mutated2)

    def test_mutate_randomness(self, random_seed):
        """Test that mutation produces different results across runs."""
        mutation = BitFlipMutation(probability=0.5)
        population = np.array([[1, 1, 1, 1, 1, 1, 1, 1]])

        results = []
        for _ in range(10):
            mutated = mutation.mutate(population)
            results.append(mutated[0].tolist())

        # Should get variety
        unique_results = set(tuple(r) for r in results)
        assert len(unique_results) > 1

    def test_mutate_with_all_zeros(self, random_seed):
        """Test mutation on all-zero population."""
        mutation = BitFlipMutation(probability=0.5)
        population = np.zeros((5, 10), dtype=int)

        mutated = mutation.mutate(population)

        # Some bits should flip to 1
        assert mutated.sum() > 0
        assert np.all((mutated == 0) | (mutated == 1))

    def test_mutate_with_all_ones(self, random_seed):
        """Test mutation on all-one population."""
        mutation = BitFlipMutation(probability=0.5)
        population = np.ones((5, 10), dtype=int)

        mutated = mutation.mutate(population)

        # Some bits should flip to 0
        assert mutated.sum() < population.sum()
        assert np.all((mutated == 0) | (mutated == 1))

    def test_probability_setter(self):
        """Test that probability can be updated."""
        mutation = BitFlipMutation(probability=0.1)
        assert mutation.probability == 0.1

        mutation.probability = 0.5
        assert mutation.probability == 0.5

    def test_str_representation(self):
        """Test string representation."""
        mutation = BitFlipMutation(probability=0.1)
        assert "BitFlipMutation" in str(mutation)
        assert "0.1" in str(mutation)

    def test_mutate_preserves_dtype(self):
        """Test that mutation preserves integer dtype."""
        mutation = BitFlipMutation(probability=0.1)
        population = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)

        mutated = mutation.mutate(population)

        # Should be integer type (int or int32/int64)
        assert mutated.dtype in [np.int32, np.int64, int]

    def test_mutate_flips_correctly(self, random_seed):
        """Test that bit flipping works correctly (0→1, 1→0)."""
        mutation = BitFlipMutation(probability=1.0)

        # Test 0 → 1
        zeros = np.zeros((2, 3), dtype=int)
        mutated_zeros = mutation.mutate(zeros)
        assert np.all(mutated_zeros == 1)

        # Test 1 → 0
        ones = np.ones((2, 3), dtype=int)
        mutated_ones = mutation.mutate(ones)
        assert np.all(mutated_ones == 0)

    def test_mutate_low_probability_changes_few_bits(self, random_seed):
        """Test that low probability mutates few bits."""
        mutation = BitFlipMutation(probability=0.01)  # 1%
        population = np.ones((10, 100), dtype=int)  # 1000 bits total

        mutated = mutation.mutate(population)

        # Expect ~10 bits to flip (1% of 1000)
        flipped = (population != mutated).sum()
        assert 1 <= flipped <= 30  # Allow variance

    def test_mutate_high_probability_changes_many_bits(self, random_seed):
        """Test that high probability mutates many bits."""
        mutation = BitFlipMutation(probability=0.9)  # 90%
        population = np.ones((10, 100), dtype=int)  # 1000 bits total

        mutated = mutation.mutate(population)

        # Expect ~900 bits to flip (90% of 1000)
        flipped = (population != mutated).sum()
        assert 800 <= flipped <= 950  # Allow variance
