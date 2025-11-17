# Feature Selection via Genetic Algorithm (FSGA)

A production-ready framework for automated feature selection using Genetic Algorithms with comprehensive evaluation and visualization tools.

## Quick Start

```bash
# Installation
git clone <repository-url>
cd feature-selection-via-genetic-algorithm
uv venv && source .venv/bin/activate
uv pip install -e .

# Run example
python experiments/run_comparison.py
```

## Basic Usage

```python
from fsga.core.genetic_algorithm import GeneticAlgorithm
from fsga.datasets.loader import load_dataset
from fsga.evaluators.accuracy_evaluator import AccuracyEvaluator
from fsga.ml.models import ModelWrapper

# Load data and setup
X_train, X_test, y_train, y_test, _ = load_dataset('iris', split=True)
model = ModelWrapper('rf', n_estimators=50, random_state=42)
evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

# Run GA
from fsga.selectors.tournament_selector import TournamentSelector
from fsga.operators.uniform_crossover import UniformCrossover
from fsga.mutations.bitflip_mutation import BitFlipMutation

ga = GeneticAlgorithm(
    num_features=X_train.shape[1],
    evaluator=evaluator,
    selector=TournamentSelector(evaluator, tournament_size=3),
    crossover_operator=UniformCrossover(),
    mutation_operator=BitFlipMutation(probability=0.01),
    population_size=50,
    num_generations=100,
    early_stopping_patience=10
)

results = ga.evolve()
print(f"Accuracy: {results['best_fitness']:.2%}")
print(f"Features: {results['best_chromosome'].sum()}/{X_train.shape[1]}")
```

## Key Features

- **Modular Design**: Swappable operators, selectors, and evaluators
- **Multiple Operators**: 5 crossover types, 5 selection strategies, 3 fitness functions
- **Baseline Comparisons**: Built-in RFE, LASSO, Mutual Information, Chi², ANOVA
- **Statistical Rigor**: Wilcoxon, Mann-Whitney, Cohen's d, Jaccard stability
- **Visualization**: 9 publication-quality plot functions
- **Experiment Framework**: `ExperimentRunner` for reproducible experiments
- **Configuration**: YAML-based configuration system

## Architecture

```
fsga/
├── core/          # GA engine (genetic_algorithm, population)
├── operators/     # Crossover: uniform, single-point, two-point, multi-point
├── mutations/     # Mutation: bitflip
├── selectors/     # Selection: tournament, roulette, ranking, elitism
├── evaluators/    # Fitness: accuracy, F1, balanced accuracy
├── ml/            # Model wrappers (sklearn integration)
├── datasets/      # Dataset loaders (iris, wine, breast_cancer, digits)
├── analysis/      # Baselines + ExperimentRunner
├── visualization/ # 9 plot functions
└── utils/         # Config, metrics, serialization, logging
```

## Documentation

- **[Getting Started](docs/GETTING_STARTED.md)** - Installation and basic usage
- **[Tutorial](docs/TUTORIAL.md)** - Step-by-step guide with examples
- **[Architecture](docs/ARCHITECTURE.md)** - System design and extension points
- **[Project Plan](PROJECT_PLAN.md)** - Remaining tasks and roadmap
- **Module READMEs** - See `fsga/*/README.md` for component details

## Example Results

**Breast Cancer Dataset** (30 features → 12 features):
- GA Accuracy: **98.3%** with **40% of features**
- All Features: 95.7% with 100% of features
- **+2.6% accuracy, 60% dimensionality reduction**

**Iris Dataset** (4 features → 2 features):
- GA Accuracy: **98.3%** with **50% of features**
- Selected: petal length, petal width

**Wine Dataset** (13 features → 6.5 features):
- GA Accuracy: **100%** with **50% of features**

## Running Experiments

```bash
# Full analysis (all datasets, all visualizations)
python experiments/run_experiment.py

# Quick test (single dataset, fewer runs)
python experiments/run_experiment.py --quick

# Specific datasets only
python experiments/run_experiment.py --datasets iris wine

# Without visualizations (faster)
python experiments/run_experiment.py --no-plots

# Results saved to: results/{mode}/{dataset}/
```

## Tests

```bash
# Run all tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=fsga --cov-report=html

# Current: 280+ tests, 82% coverage
```

## Configuration

Example config (`configs/default.yaml`):

```yaml
population_size: 50
num_generations: 100
mutation_rate: 0.01
crossover_rate: 0.8
early_stopping_patience: 10

dataset:
  name: iris
  split_ratio: 0.7
```

Load with:

```python
from fsga.utils.config import Config
config = Config.from_file('configs/default.yaml')
```

## Citation

If you use this framework in research, please cite:

```bibtex
@software{fsga2025,
  title={Feature Selection via Genetic Algorithm},
  author={Piotr Krzysztof Lis},
  year={2025},
  url={https://github.com/straightchlorine/feature-selection-via-genetic-algorithm}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! See module READMEs for extension points:
- New operators: `fsga/operators/README.md`
- New selectors: `fsga/selectors/README.md`
- New evaluators: `fsga/evaluators/README.md`
