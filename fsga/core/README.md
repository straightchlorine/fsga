# fsga.core - Genetic Algorithm Engine

Core GA orchestration and population management.

## Components

### `genetic_algorithm.py`
Main GA orchestrator - coordinates selection, crossover, mutation, and fitness evaluation.

**Adapted from**: `knapsack/genetic_algorithm.py`

**Key Changes from Knapsack**:
- Removed dataset-specific logic (weights, capacity)
- Added ML model training integration
- Added early stopping based on convergence
- Support for multi-objective optimization

**Usage**:
```python
from fsga.core.genetic_algorithm import GeneticAlgorithm
from fsga.evaluators.accuracy_evaluator import AccuracyEvaluator

ga = GeneticAlgorithm(
    num_features=30,  # Number of features in dataset
    evaluator=evaluator,
    selector=selector,
    crossover_operator=crossover,
    mutation_operator=mutation,
    population_size=50,
    num_generations=100,
    early_stopping_patience=10
)

results = ga.evolve()
print(f"Best fitness: {results['best_fitness']}")
print(f"Best chromosome: {results['best_chromosome']}")
```

### `population.py`
Population management - initialization, diversity tracking, parent selection.

**Adapted from**: `knapsack/population.py`

**Initialization Strategies**:
- `random`: Uniform random bits
- `correlation_biased`: Favor features correlated with target
- `mutual_info`: Based on mutual information scores
- `uniform`: Equal probability for 0/1

**Usage**:
```python
from fsga.core.population import Population

pop = Population(
    num_features=30,
    population_size=50,
    selector=selector
)

pop.initialize_population(strategy='correlation_biased', X=X_train, y=y_train)
diversity = pop.measure_diversity()  # Returns percentage of unique chromosomes
```

### `chromosome.py`
Chromosome representation and utility functions.

**Purpose**: Binary array where `chromosome[i]=1` means "include feature i"

**Usage**:
```python
from fsga.core.chromosome import Chromosome

chrom = Chromosome.random(num_features=10)  # Random binary array
selected_features = chrom.selected_indices()  # Indices where value is 1
num_features = chrom.count_selected()  # Number of 1s
```

## Design Philosophy

- **Modularity**: Core engine is agnostic to fitness function
- **Extensibility**: Easy to swap operators, selectors, evaluators
- **Performance**: Optimized loops, minimal copying

## Extending

To add a new initialization strategy:

```python
# In population.py
def _init_custom_strategy(self, **kwargs):
    """Your custom initialization logic."""
    chromosomes = []
    for _ in range(self.population_size):
        # Generate chromosome based on your strategy
        chromosome = your_logic_here()
        chromosomes.append(chromosome)
    return np.array(chromosomes)

# Register in initialize_population()
strategies = {
    'random': self._init_random,
    'custom': self._init_custom_strategy  # Add here
}
```
