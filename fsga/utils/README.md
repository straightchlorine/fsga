# fsga.utils - Utilities

Configuration, metrics, logging, and serialization utilities.

## Components

### `config.py`
YAML configuration management for reproducible experiments.

```python
from fsga.utils.config import Config, load_config

# Load from YAML
config = load_config('configs/default.yaml')

# Access nested values
pop_size = config.get('genetic_algorithm.population_size')  # 50
model_type = config.get('model.type')  # 'rf'

# With defaults
mutation_rate = config.get('genetic_algorithm.mutation_rate', default=0.01)

# Save config
config.save('configs/experiment_run_2025-10-06.yaml')
```

**Example YAML**:
```yaml
# configs/default.yaml
genetic_algorithm:
  population_size: 50
  num_generations: 100
  mutation_rate: 0.01
  crossover_rate: 0.7
  early_stopping_patience: 10

model:
  type: rf
  params:
    n_estimators: 100
    max_depth: 10
    random_state: 42

dataset:
  name: wine
  test_size: 0.2
  validation_size: 0.1
  stratify: true
  random_state: 42

logging:
  level: INFO
  log_file: results/logs/experiment.log
```

### `metrics.py`
Performance metrics calculation and aggregation.

```python
from fsga.utils.metrics import MetricsCalculator

calc = MetricsCalculator()

# Single metrics
accuracy = calc.accuracy(y_true, y_pred)
f1 = calc.f1_score(y_true, y_pred, average='weighted')
balanced_acc = calc.balanced_accuracy(y_true, y_pred)

# All metrics at once
metrics = calc.calculate_all(y_true, y_pred)
# Returns: {'accuracy': 0.92, 'f1': 0.91, 'precision': 0.93, ...}

# Feature selection metrics
stability = calc.jaccard_stability(selected_features_list)
sparsity = calc.feature_sparsity(chromosome)  # % features not selected
```

**Available Metrics**:
- Classification: accuracy, precision, recall, F1, balanced accuracy
- Feature Selection: stability (Jaccard), sparsity, selection frequency
- GA-specific: convergence rate, diversity loss, selection pressure

### `logging_setup.py`
Structured logging for experiments.

```python
from fsga.utils.logging_setup import setup_logging

# Setup logger
logger = setup_logging(
    name='fsga',
    log_file='results/logs/experiment.log',
    level='INFO',  # DEBUG, INFO, WARNING, ERROR
    console=True   # Also print to console
)

# Use in code
logger.info("Starting GA evolution...")
logger.debug(f"Generation {i}: best fitness = {fitness}")
logger.warning("Population diversity < 0.1, risk of premature convergence")
logger.error("Fitness evaluation failed!")
```

**Log Format**:
```
2025-10-06 14:32:15 | INFO | fsga.core.genetic_algorithm | Starting evolution with pop_size=50
2025-10-06 14:32:16 | DEBUG | fsga.evaluators.accuracy_evaluator | Evaluating chromosome [1,0,1,1,0...]
2025-10-06 14:32:45 | INFO | fsga.core.genetic_algorithm | Converged at generation 73
```

### `serialization.py`
Save and load experiment results.

```python
from fsga.utils.serialization import save_results, load_results

# Save results
save_results(
    results,
    filepath='results/wine_experiment.pkl',
    format='pickle'  # or 'json'
)

# Load results
loaded_results = load_results('results/wine_experiment.pkl')

# Save multiple runs
save_results(
    {'run_1': results1, 'run_2': results2, ...},
    filepath='results/all_runs.pkl'
)
```

**Serialization Formats**:
- `pickle`: Fast, Python-specific, supports complex objects (default)
- `json`: Human-readable, language-agnostic, limited types
- `yaml`: Human-readable, good for configs

**What to Save**:
```python
experiment_data = {
    'config': config.to_dict(),
    'results': {
        'best_chromosome': best_chromosome,
        'best_fitness': best_fitness,
        'fitness_history': fitness_history,
        'diversity_history': diversity_history,
        'selected_features': selected_feature_names
    },
    'metadata': {
        'timestamp': datetime.now(),
        'runtime_seconds': runtime,
        'dataset': 'wine',
        'model': 'RandomForest'
    }
}

save_results(experiment_data, 'results/experiment_2025-10-06.pkl')
```

## Helper Functions

### `random_seed.py`
Reproducibility utilities.

```python
from fsga.utils.random_seed import set_all_seeds

# Set all random seeds at once
set_all_seeds(42)
# Sets: numpy.random, random, sklearn, torch (if available)

# Now all runs are reproducible
```

### `timer.py`
Execution timing (ported from knapsack).

```python
from fsga.utils.timer import Timer

with Timer() as t:
    ga.evolve()

print(f"Evolution took {t.elapsed_ms:.2f} ms")
```

### `validation.py`
Input validation utilities.

```python
from fsga.utils.validation import validate_chromosome, validate_dataset

# Validate chromosome
validate_chromosome(chromosome, expected_length=30)
# Raises ValueError if not binary or wrong length

# Validate dataset
validate_dataset(X, y, min_samples=50)
# Raises ValueError if shapes don't match or too few samples
```

## Configuration Best Practices

### Config Versioning
```python
# configs/v1_baseline.yaml
version: "1.0"
description: "Baseline GA configuration"
# ... params

# configs/v2_adaptive.yaml
version: "2.0"
description: "GA with adaptive operators"
# ... params
```

### Environment-Specific Configs
```python
# configs/development.yaml (small, fast)
genetic_algorithm:
  population_size: 10
  num_generations: 20

# configs/production.yaml (large, thorough)
genetic_algorithm:
  population_size: 100
  num_generations: 500
```

Load based on environment:
```python
import os
env = os.getenv('ENV', 'development')
config = load_config(f'configs/{env}.yaml')
```

## Logging Best Practices

### Log Levels
- **DEBUG**: Detailed info for debugging (e.g., every chromosome evaluation)
- **INFO**: General progress (e.g., generation completed)
- **WARNING**: Potential issues (e.g., low diversity)
- **ERROR**: Failures (e.g., fitness evaluation crashed)

### What to Log
```python
# DON'T: Too verbose
for i, chrom in enumerate(population):
    logger.debug(f"Chromosome {i}: {chrom}")  # 50 log lines per generation!

# DO: Aggregate
logger.info(f"Generation {gen}: best={best_fit:.4f}, avg={avg_fit:.4f}, diversity={div:.2f}%")
```

## Serialization Best Practices

### File Organization
```
results/
├── logs/
│   └── experiment_2025-10-06.log
├── checkpoints/
│   ├── gen_10.pkl  # Save every N generations
│   ├── gen_20.pkl
│   └── gen_30.pkl
└── final/
    ├── wine_ga_results.pkl
    └── wine_comparison.json  # Human-readable summary
```

### Checkpoint Saving
```python
# In genetic_algorithm.py
if generation % 10 == 0:  # Every 10 generations
    checkpoint = {
        'generation': generation,
        'population': self.population.chromosomes,
        'best_fitness': self.best_fitness,
        'fitness_history': self.fitness_history
    }
    save_results(checkpoint, f'results/checkpoints/gen_{generation}.pkl')
```

## Extending

Add custom metric:

```python
# In metrics.py
class MetricsCalculator:
    def my_custom_metric(self, y_true, y_pred):
        """Your custom metric calculation."""
        # Calculation logic
        return score

    def calculate_all(self, y_true, y_pred):
        return {
            'accuracy': self.accuracy(y_true, y_pred),
            'f1': self.f1_score(y_true, y_pred),
            'my_custom': self.my_custom_metric(y_true, y_pred)  # Add here
        }
```
