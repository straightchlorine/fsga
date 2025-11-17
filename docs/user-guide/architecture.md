# FSGA Architecture

System design and implementation overview.

## System Components

```
fsga/
├── core/          # GA Engine
│   ├── genetic_algorithm.py  # Main GA loop
│   └── population.py          # Population management
│
├── operators/     # Crossover Operators (5 types)
│   ├── uniform_crossover.py
│   ├── single_point_crossover.py
│   ├── two_point_crossover.py
│   └── multi_point_crossover.py
│
├── mutations/     # Mutation Operators
│   └── bitflip_mutation.py
│
├── selectors/     # Selection Strategies (5 types)
│   ├── tournament_selector.py
│   ├── roulette_selector.py
│   ├── ranking_selector.py
│   └── elitism_selector.py
│
├── evaluators/    # Fitness Functions (3 types)
│   ├── accuracy_evaluator.py
│   ├── f1_evaluator.py
│   └── balanced_accuracy_evaluator.py
│
├── ml/            # ML Integration
│   └── models.py              # ModelWrapper for sklearn
│
├── datasets/      # Dataset Loaders
│   └── loader.py              # load_dataset()
│
├── analysis/      # Experiment Framework
│   ├── baselines.py           # 6 baseline methods
│   └── experiment_runner.py   # ExperimentRunner class
│
├── visualization/ # Plotting (9 functions)
│   └── plots.py
│
└── utils/         # Utilities
    ├── config.py              # Configuration management
    ├── metrics.py             # Statistical tests
    └── serialization.py       # Results I/O
```

## Design Patterns

### Strategy Pattern

All operators use the Strategy pattern for swappable implementations:

```python
# Abstract base class
class Crossover(ABC):
    @abstractmethod
    def crossover(self, parent1, parent2):
        pass

# Concrete implementations
class UniformCrossover(Crossover):
    def crossover(self, parent1, parent2):
        # Implementation...
```

This allows easy switching:

```python
ga = GeneticAlgorithm(
    crossover_operator=UniformCrossover(),  # or SinglePointCrossover()
    # ...
)
```

### Dependency Injection

Components are injected, not hardcoded:

```python
ga = GeneticAlgorithm(
    evaluator=evaluator,           # Inject fitness function
    selector=selector,             # Inject selection strategy
    crossover_operator=crossover,  # Inject crossover
    mutation_operator=mutation     # Inject mutation
)
```

## Data Flow

```
1. Initialization
   Dataset → load_dataset() → X_train, y_train
   Model → ModelWrapper → sklearn classifier
   Evaluator → AccuracyEvaluator(X, y, model)

2. GA Loop (per generation)
   Population → Selector.select() → Parents
   Parents → Crossover.crossover() → Offspring
   Offspring → Mutation.mutate() → Mutated Offspring
   Mutated → Evaluator.evaluate() → Fitness Scores

3. Selection Pressure
   Fitness Scores → Elitism/Tournament → Next Generation

4. Convergence Check
   if no improvement for N generations:
       early_stopping_patience → STOP

5. Results
   Best Chromosome → Feature Subset → Performance Metrics
```

## Key Classes

### GeneticAlgorithm

Main orchestrator with early stopping and metrics tracking.

**Properties:**
- `population_size`, `num_generations`
- `mutation_rate`, `crossover_rate`
- `early_stopping_patience`

**Methods:**
- `evolve()` - Main GA loop
- `_initialize_population()` - Create initial population
- `_selection()` - Select parents
- `_crossover()` - Apply crossover
- `_mutation()` - Apply mutation
- `_evaluate_population()` - Calculate fitness

### ExperimentRunner

Framework for reproducible experiments.

**Methods:**
- `run_ga_experiment(n_runs=10)` - Multiple GA runs
- `run_baseline_experiment(method)` - Run baseline method
- `compare_methods()` - Statistical comparison
- `generate_summary_report()` - Text report

## Extension Points

### Adding a New Crossover Operator

```python
from fsga.operators.crossover import Crossover
import numpy as np

class MyCustomCrossover(Crossover):
    def crossover(self, parent1, parent2):
        # Your crossover logic
        child1 = ...
        child2 = ...
        return child1, child2
```

### Adding a New Evaluator

```python
from fsga.evaluators.evaluator import Evaluator

class MyCustomEvaluator(Evaluator):
    def evaluate(self, chromosome):
        # Select features
        selected_features = np.where(chromosome == 1)[0]
        X_selected = self.X_train[:, selected_features]

        # Train and evaluate
        self.model.fit(X_selected, self.y_train)
        score = self.model.score(X_selected, self.y_train)

        return score
```

## Performance Considerations

### Fitness Evaluation Bottleneck

Training ML models is expensive. Strategies:
- Use validation set (not full training set)
- Fast models (RandomForest > SVM)
- Smaller population (50 vs 100)

### Memory Usage

Each chromosome: `num_features` bytes
Population: `population_size × num_features × generations` bytes

For 30 features, 50 population, 100 generations:
= 50 × 30 × 100 = 150KB (negligible)

## Configuration

### YAML Config Files

```yaml
# configs/default.yaml
population_size: 50
num_generations: 100
mutation_rate: 0.01
crossover_rate: 0.8
early_stopping_patience: 10

dataset:
  name: iris
  split_ratio: 0.7

model:
  type: rf
  n_estimators: 50
  random_state: 42
```

### Loading Configs

```python
from fsga.utils.config import Config

config = Config.from_file('configs/default.yaml')
population_size = config.population_size
```

## Testing Strategy

- **Unit tests**: Each operator in isolation
- **Integration tests**: Full GA workflow
- **Edge case tests**: Empty features, single sample
- **Coverage**: 82% code coverage

See module READMEs for detailed developer guides:
- `fsga/core/README_DEV.md` - Core GA internals
- `fsga/operators/README.md` - Crossover operators
- `fsga/selectors/README.md` - Selection strategies
- `fsga/evaluators/README.md` - Fitness functions
