# Quick Start

Get up and running with FSGA in minutes.

## Installation

```bash
pip install fsga
```

## Basic Usage

### 1. Simple Feature Selection

```python
from fsga.core.genetic_algorithm import GeneticAlgorithm
from fsga.datasets.loader import load_dataset
from fsga.evaluators.accuracy_evaluator import AccuracyEvaluator
from fsga.ml.models import ModelWrapper
from fsga.selectors.tournament_selector import TournamentSelector
from fsga.operators.uniform_crossover import UniformCrossover
from fsga.mutations.bitflip_mutation import BitFlipMutation

# Load dataset
X_train, X_test, y_train, y_test, feature_names = load_dataset('iris', split=True)

# Setup ML model
model = ModelWrapper('rf', n_estimators=50, random_state=42)

# Create evaluator (fitness function)
evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

# Create GA components
selector = TournamentSelector(evaluator, tournament_size=3)
crossover = UniformCrossover()
mutation = BitFlipMutation(probability=0.01)

# Initialize and run GA
ga = GeneticAlgorithm(
    num_features=X_train.shape[1],
    evaluator=evaluator,
    selector=selector,
    crossover_operator=crossover,
    mutation_operator=mutation,
    population_size=50,
    num_generations=100,
    early_stopping_patience=10,
    verbose=True
)

results = ga.evolve()

# Print results
print(f"Best Accuracy: {results['best_fitness']:.4f}")
print(f"Features Selected: {results['best_chromosome'].sum()}/{X_train.shape[1]}")
print(f"Selected Features: {[feature_names[i] for i in range(len(results['best_chromosome'])) if results['best_chromosome'][i] == 1]}")
```

### 2. Using Configuration Files

```python
from fsga.utils.config import Config

# Load pre-defined configuration
config = Config.from_file('configs/default.yaml')

# Access settings
population_size = config.population_size
dataset_name = config.get('dataset.name')
```

### 3. Running Experiments

```python
from fsga.analysis.experiment_runner import ExperimentRunner

# Initialize experiment runner
runner = ExperimentRunner(
    dataset_name='iris',
    model_type='rf',
    n_runs=10,
    random_state=42
)

# Run GA
runner.run_ga_experiment(
    population_size=50,
    num_generations=100,
    verbose=True
)

# Run baseline methods
runner.run_baseline_experiment('rfe', verbose=True)
runner.run_baseline_experiment('lasso', verbose=True)
runner.run_all_features_baseline()

# Compare methods statistically
comparisons = runner.compare_methods(verbose=True)

# Generate report
print(runner.generate_summary_report())
```

### 4. Visualization

```python
from fsga.visualization import (
    plot_fitness_evolution,
    plot_method_comparison,
    plot_multi_metric_comparison
)

# Plot fitness over generations
plot_fitness_evolution(
    results['fitness_history'],
    save_path='fitness_evolution.png'
)

# Compare multiple methods
method_results = {
    'GA': runner.results['GA']['accuracies'],
    'RFE': runner.results['RFE']['accuracies'],
    'LASSO': runner.results['LASSO']['accuracies']
}

plot_method_comparison(
    method_results,
    metric_name='Accuracy',
    save_path='method_comparison.png'
)

# Multi-metric comparison
plot_multi_metric_comparison(
    runner.results,
    save_path='multi_metric.png'
)
```

## Command-Line Usage

### Run Experiments

```bash
# Full analysis
python experiments/run_experiment.py

# Quick test
python experiments/run_experiment.py --quick

# Specific datasets
python experiments/run_experiment.py --datasets iris wine

# Custom runs
python experiments/run_experiment.py --runs 20 --no-plots
```

## Next Steps

- **[Tutorial](tutorial.md)** - Detailed walkthrough with examples
- **[User Guide](../user-guide/architecture.md)** - Architecture and design
- **[API Reference](../api/core.md)** - Complete API documentation
- **[Examples](../examples/basic.md)** - More code examples
