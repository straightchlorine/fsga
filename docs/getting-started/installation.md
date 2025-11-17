# Getting Started with FSGA

Quick start guide for Feature Selection via Genetic Algorithm.

## Installation

```bash
cd ~/code/feature-selection-via-genetic-algorithm
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Basic Usage

### Simple Example

```python
from fsga.core.genetic_algorithm import GeneticAlgorithm
from fsga.datasets.loader import load_dataset
from fsga.evaluators.accuracy_evaluator import AccuracyEvaluator
from fsga.ml.models import ModelWrapper
from fsga.operators.uniform_crossover import UniformCrossover
from fsga.mutations.bitflip_mutation import BitFlipMutation
from fsga.selectors.tournament_selector import TournamentSelector

# Load data
X_train, X_test, y_train, y_test, names = load_dataset('iris', split=True)

# Setup components
model = ModelWrapper('rf', n_estimators=50, random_state=42)
evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)
selector = TournamentSelector(evaluator, tournament_size=3)
crossover = UniformCrossover()
mutation = BitFlipMutation(probability=0.01)

# Run GA
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
print(f"Best accuracy: {results['best_fitness']:.4f}")
print(f"Features selected: {results['best_chromosome'].sum()}/{X_train.shape[1]}")
```

## Using Configuration Files

```python
from fsga.utils.config import Config
from fsga.core.genetic_algorithm import GeneticAlgorithm

# Load config
config = Config.from_file('configs/default.yaml')

# Use config parameters
ga = GeneticAlgorithm(
    num_features=config.get('num_features'),
    population_size=config.population_size,
    num_generations=config.num_generations,
    # ...
)
```

## Running Experiments

### Quick Comparison

```python
from fsga.analysis.experiment_runner import ExperimentRunner

# Run GA vs baselines on Iris dataset
runner = ExperimentRunner(dataset_name='iris', n_runs=10)
runner.run_ga_experiment(verbose=True)
runner.run_baseline_experiment('rfe', verbose=True)
runner.run_all_features_baseline()

# Statistical comparison
comparisons = runner.compare_methods(verbose=True)
print(runner.generate_summary_report())
```

### Using Experiment Scripts

```bash
# Full analysis (3 datasets, 4 baselines, all plots)
python experiments/run_experiment.py

# Quick test mode (faster)
python experiments/run_experiment.py --quick

# Custom configuration
python experiments/run_experiment.py --datasets iris --runs 5 --no-plots
```

## Visualization

```python
from fsga.visualization import (
    plot_fitness_evolution,
    plot_method_comparison,
    plot_multi_metric_comparison
)

# Fitness over generations
plot_fitness_evolution(
    results['fitness_history'],
    save_path='results/fitness.png'
)

# Compare methods
plot_method_comparison(
    runner.results,
    save_path='results/comparison.png'
)

# Multi-metric dashboard
plot_multi_metric_comparison(
    runner.results,
    save_path='results/multi_metric.png'
)
```

## Next Steps

- See [TUTORIAL.md](TUTORIAL.md) for detailed walkthrough
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- See [API_REFERENCE.md](API_REFERENCE.md) for complete API docs
- See module READMEs in `fsga/*/README.md` for component details
