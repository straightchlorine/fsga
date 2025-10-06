# fsga.selectors - Selection Strategies

Methods for choosing parents to create the next generation.

## Components

### `selector.py`
Abstract base class.

```python
class Selector(ABC):
    @abstractmethod
    def select(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns two parent chromosomes."""
        pass
```

### Implemented Selectors

#### `tournament_selector.py`
Select `k` random individuals, pick the best.

**Ported from**: `knapsack/selectors/tournament_selector.py`

```python
selector = TournamentSelector(evaluator, tournament_size=3)
parent1, parent2 = selector.select()
```

**Selection Pressure**: tournament_size ↑ = more pressure (favors best)

#### `roulette_selector.py`
Probability proportional to fitness (fitness wheel).

**Ported from**: `knapsack/selectors/roulette_selector.py`

**Warning**: Can be dominated by very fit individuals (premature convergence)

#### `ranking_selector.py`
Rank population, select based on rank (not raw fitness).

**Ported from**: `knapsack/selectors/ranking_selector.py`

**Advantage**: More stable than roulette (outliers don't dominate)

#### `elitism_selector.py`
Always select the top `n` individuals.

**Ported from**: `knapsack/selectors/elitism_selector.py`

**Usage**: Combine with other selectors for "elitism + tournament"

#### `random_selector.py`
Uniformly random selection.

**Use case**: Baseline comparison, high exploration

#### `nsga2_selector.py` ✨ NEW
Non-dominated sorting for multi-objective optimization.

**Objectives**: Maximize accuracy AND minimize features

```python
selector = NSGA2Selector(multi_objective_evaluator)
# Returns Pareto-optimal parents
```

**Output**: Population sorted by Pareto rank, then crowding distance

## Comparison

| Selector | Exploration | Exploitation | Premature Convergence Risk |
|----------|-------------|--------------|----------------------------|
| Random | Highest | Lowest | None |
| Roulette | Low | High | High |
| Tournament | Medium | Medium | Medium |
| Ranking | Medium | Medium | Low |
| Elitism | Lowest | Highest | Highest |
| NSGA-II | Medium | Medium | Low (maintains diversity) |

## Best Practices

**For single-objective**:
- Start: Tournament (size=3) for balanced exploration/exploitation
- Converged: Elitism (preserve best solutions)

**For multi-objective**:
- Use NSGA-II (handles trade-offs between accuracy and sparsity)

## Extending

```python
from fsga.selectors.selector import Selector

class MySelector(Selector):
    def select(self):
        # Your selection logic
        indices = your_selection_method(self.population, self.evaluator)
        return self.population[indices[0]], self.population[indices[1]]
```
