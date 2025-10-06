# fsga.operators - Crossover Operators

Recombination operators for creating offspring from parent chromosomes.

## Components

### `crossover.py`
Abstract base class defining the crossover interface.

```python
class Crossover(ABC):
    @abstractmethod
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Returns array of 2 children."""
        pass
```

### Implemented Operators

#### `uniform_crossover.py`
Each gene randomly chosen from either parent (50/50 probability).

**Ported from**: `knapsack/operators/uniform_crossover.py`

**Example**:
```
Parent 1: [1, 1, 0, 1, 0]
Parent 2: [0, 1, 1, 0, 0]
Mask:     [T, F, T, F, T]
Child 1:  [1, 1, 1, 0, 0]  (T → parent1, F → parent2)
Child 2:  [0, 1, 0, 1, 0]  (opposite)
```

#### `single_point_crossover.py`
Split at random point, swap tails.

**Example**:
```
Parent 1: [1, 1 | 0, 1, 0]  (split at index 2)
Parent 2: [0, 1 | 1, 0, 0]
Child 1:  [1, 1, 1, 0, 0]
Child 2:  [0, 1, 0, 1, 0]
```

#### `two_point_crossover.py`
Two split points, swap middle segment.

#### `adaptive_crossover.py` ✨ NEW
Adjusts crossover behavior based on population diversity.

**Innovation**: Low diversity → more disruptive crossover (exploration)

```python
crossover = AdaptiveCrossover()
children = crossover.crossover(parent1, parent2, population_diversity=0.3)
# Uses aggressive crossover when diversity is low
```

## Usage

```python
from fsga.operators.uniform_crossover import UniformCrossover

crossover = UniformCrossover()
parent1 = np.array([1, 0, 1, 0, 1])
parent2 = np.array([0, 1, 1, 1, 0])

children = crossover.crossover(parent1, parent2)
# Returns shape (2, 5) - two children
```

## Comparison

| Operator | Exploration | Exploitation | Best For |
|----------|-------------|--------------|----------|
| Uniform | High | Low | Early generations, high-dim |
| Single-Point | Medium | Medium | Balanced search |
| Two-Point | Medium | High | Preserving feature blocks |
| Adaptive | Dynamic | Dynamic | Long runs, adaptive search |

## Extending

Create new operator:

```python
from fsga.operators.crossover import Crossover

class MyCustomCrossover(Crossover):
    def crossover(self, parent1, parent2):
        # Your recombination logic
        child1 = ...
        child2 = ...
        return np.array([child1, child2])
```
