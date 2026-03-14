# Getting Started

## Installation

```bash
git clone https://github.com/straightchlorine/feature-selection-via-genetic-algorithm.git
cd feature-selection-via-genetic-algorithm
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Basic Usage

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

# Setup
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
print(f"Accuracy: {results['best_fitness']:.4f}")
print(f"Features: {results['best_chromosome'].sum()}/{X_train.shape[1]}")
```

## Configuration Files

```python
from fsga.utils.config import Config

config = Config.from_file('configs/default.yaml')
population_size = config.population_size
dataset_name = config.get('dataset.name')
```

## Running Experiments

```bash
# Full analysis (3 datasets, baselines, plots)
python experiments/run_experiment.py

# Quick test
python experiments/run_experiment.py --quick

# Specific datasets
python experiments/run_experiment.py --datasets iris wine

# Custom runs, no plots
python experiments/run_experiment.py --runs 20 --no-plots
```

Or programmatically:

```python
from fsga.analysis.experiment_runner import ExperimentRunner

runner = ExperimentRunner(dataset_name='iris', model_type='rf', n_runs=10, random_state=42)
runner.run_ga_experiment(population_size=50, num_generations=100, verbose=True)
runner.run_baseline_experiment('rfe', verbose=True)
runner.run_baseline_experiment('lasso', verbose=True)
runner.run_all_features_baseline()

comparisons = runner.compare_methods(verbose=True)
print(runner.generate_summary_report())
```

## Visualization

```python
from fsga.visualization import (
    plot_fitness_evolution,
    plot_method_comparison,
    plot_multi_metric_comparison
)

# Fitness over generations
plot_fitness_evolution(results['fitness_history'], save_path='fitness.png')

# Compare methods
plot_method_comparison(runner.results, save_path='comparison.png')

# Multi-metric dashboard
plot_multi_metric_comparison(runner.results, save_path='multi_metric.png')
```

## Next Steps

- [Tutorial](tutorial.md) -- detailed walkthrough with examples
- [Architecture](../user-guide/architecture.md) -- system design
- [Project Plan](../about/project-plan.md) -- status and roadmap
