# fsga.mutations - Mutation Operators

Operators for introducing variation into offspring.

## Components

### `mutation.py`
Abstract base class.

```python
class Mutation(ABC):
    def __init__(self, probability: float):
        self.probability = probability

    @abstractmethod
    def mutate(self, population: np.ndarray) -> np.ndarray:
        pass
```

### Implemented Operators

#### `bitflip_mutation.py`
Flip each bit with probability `p`.

**Ported from**: `knapsack/mutations/bitflip_mutation.py`

```python
mutation = BitFlipMutation(probability=0.01)
offspring = mutation.mutate(offspring)
# Each gene has 1% chance of flipping (0→1 or 1→0)
```

#### `gaussian_mutation.py`
Add Gaussian noise, threshold to binary.

**Ported from**: `knapsack/mutations/gaussian_mutation.py`

#### `dynamic_mutation.py`
Mutation rate decreases over generations.

**Ported from**: `knapsack/mutations/dynamic_mutation.py`

**Rationale**: High exploration early, low perturbation late

```python
mutation = DynamicMutation(initial_prob=0.1, final_prob=0.001, max_generations=100)
# Generation 0: prob = 0.1
# Generation 50: prob = 0.05
# Generation 100: prob = 0.001
```

#### `feature_aware_mutation.py` ✨ NEW
Higher mutation rate for correlated features.

**Innovation**: If features are correlated, swapping one for another is "safe"

```python
mutation = FeatureAwareMutation(
    probability=0.01,
    feature_correlation_matrix=correlation_matrix
)
offspring = mutation.mutate(offspring)
# Features with correlation > 0.8 have 5x mutation rate
```

## Usage

```python
from fsga.mutations.bitflip_mutation import BitFlipMutation

mutation = BitFlipMutation(probability=0.01)

# Mutate population of 50 chromosomes
offspring = np.array([[1, 0, 1, ...], ...])  # Shape: (50, num_features)
mutated = mutation.mutate(offspring)
```

## Mutation Rate Guidelines

| Dataset Size | Features | Suggested Rate |
|--------------|----------|----------------|
| Small (< 100) | < 50 | 0.05 - 0.1 |
| Medium (100-1000) | 50-500 | 0.01 - 0.05 |
| Large (> 1000) | > 500 | 0.001 - 0.01 |

**Rule of thumb**: `mutation_rate ≈ 1 / num_features`

## Extending

```python
from fsga.mutations.mutation import Mutation

class MyMutation(Mutation):
    def mutate(self, population):
        mutated = population.copy()
        for i in range(len(mutated)):
            for j in range(len(mutated[i])):
                if np.random.rand() < self.probability:
                    mutated[i][j] = your_mutation_logic()
        return mutated
```
