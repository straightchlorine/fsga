"""Bit-flip mutation operator.

Ported directly from knapsack-problem/knapsack/mutations/bitflip_mutation.py
No changes needed - works identically for feature selection.
"""

import numpy as np

from fsga.mutations.mutation import Mutation


class BitFlipMutation(Mutation):
    """Bit-flip mutation: flips each bit with probability p.

    For each gene in each chromosome, independently flips 0→1 or 1→0
    with probability equal to the mutation rate.

    Example:
        Chromosome: [1, 0, 1, 0, 1]
        Probability: 0.2 (20% chance per gene)

        Mutation mask: [F, T, F, F, T]  (random, ~20% True)
        Result:        [1, 1, 1, 0, 0]  (bits at positions 1 and 4 flipped)

    Properties:
        - Simple and effective
        - Maintains binary constraint
        - Standard mutation for binary GAs

    Usage:
        >>> mutation = BitFlipMutation(probability=0.01)
        >>> offspring = np.array([[1, 0, 1], [0, 1, 0]])
        >>> mutated = mutation.mutate(offspring)
    """

    def mutate(self, population: np.ndarray, generation: int = 0) -> np.ndarray:
        """Apply bit-flip mutation to population.

        Args:
            population: Population of chromosomes (2D array)
            generation: Current generation (unused, for interface compatibility)

        Returns:
            np.ndarray: Mutated population (same shape as input)

        Note:
            Does not modify input population (creates copy).
        """
        mutated_population = population.copy()
        mutated_population = np.array(mutated_population, dtype=int)

        # Generate mutation mask (True where mutation should occur)
        mutation_mask = np.random.rand(*mutated_population.shape) < self.probability

        # Flip bits using XOR (0 XOR 1 = 1, 1 XOR 1 = 0)
        mutated_population[mutation_mask] ^= 1

        return mutated_population
