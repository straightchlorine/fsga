---
title: "Architecture: Package Layout and Extension Points"
description: "How FSGA is put together: package layout, the strategy pattern and dependency injection, data flow, key classes, extension points and the YAML config."
---

# FSGA Architecture

System design and implementation overview.

## System Components

```
fsga/
├── core/          # GA Engine
│   ├── genetic_algorithm.py  # Main GA loop
│   └── population.py          # Population management
│
├── operators/     # Crossover Operators (4 types)
│   ├── uniform_crossover.py
│   ├── single_point_crossover.py
│   ├── two_point_crossover.py
│   └── multi_point_crossover.py
│
├── mutations/     # Mutation Operators
│   └── bitflip_mutation.py
│
├── selectors/     # Selection Strategies (4 types)
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
│   ├── baselines.py           # 5 baseline methods
│   └── experiment_runner.py   # ExperimentRunner class
│
├── visualization/ # Plotting (9 functions)
│   └── plots.py
│
└── utils/         # Utilities
    ├── config.py              # Configuration management
    ├── metrics.py             # Statistical tests
    ├── logging.py             # Logging helpers
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
        pass
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
    num_features=num_features,
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
   Evaluator → AccuracyEvaluator(X_train, y_train, X_val, y_val, model)

2. GA Loop (per generation)
   Population → Selector.select() → Parents
   Parents → Crossover.crossover() → Offspring
   Offspring → Mutation.mutate() → Mutated Offspring
   Mutated → Evaluator.evaluate() → Fitness Scores

3. Selection Pressure
   Fitness Scores → Elitism/Tournament → Parents of Next Generation

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
- `population_size`, `generations`, `mutation_rate`
- `early_stopping_patience` and `num_features` are constructor arguments

**Methods:**
- `evolve()` - Main GA loop
- `new_generation(current_generation)` - Build the next population
- `reinitialize_population()` - Start again from a new random population
- `get_best_solution(n=1)` - Best chromosome(s) in the current population
- `clear_metrics()` - Reset the tracked metrics

### ExperimentRunner

Framework for reproducible experiments.

**Methods:**
- `run_ga_experiment(population_size, num_generations, ...)` - Multiple GA runs (`n_runs` is set in the constructor)
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
        return np.array([child1, child2])  # shape (2, num_features)
```

### Adding a New Evaluator

```python
from fsga.evaluators.evaluator import Evaluator
import numpy as np

class MyCustomEvaluator(Evaluator):
    def __init__(self, X_train, y_train, X_val, y_val, model):
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.model = model

    def evaluate(self, chromosome):
        # Select features
        selected_features = np.where(chromosome == 1)[0]
        X_train_sel = self.X_train[:, selected_features]
        X_val_sel = self.X_val[:, selected_features]

        # Train on the training set, score on the validation set
        self.model.fit(X_train_sel, self.y_train)
        score = self.model.score(X_val_sel, self.y_val)

        return score
```

## Performance Considerations

### Fitness Evaluation Bottleneck

Training ML models is expensive. Strategies:
- Use validation set (not full training set)
- Fast models (SVM > RandomForest)
- Smaller population (50 vs 100)

### Memory Usage

Each chromosome: `num_features` integers (8 bytes each with the default int64)
Population: `population_size × num_features` genes (only the current population is kept)

For 30 features, 50 population:
= 50 × 30 × 8 = 12KB (negligible)

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
  test_size: 0.2
  val_size: 0.1

model:
  type: rf
  n_estimators: 100
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
- **Coverage**: about 67% line coverage

See module READMEs for detailed developer guides:
- `fsga/core/README_DEV.md` - Core GA internals
- `fsga/operators/README.md` - Crossover operators
- `fsga/selectors/README.md` - Selection strategies
- `fsga/evaluators/README.md` - Fitness functions
