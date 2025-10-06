# Feature Selection via Genetic Algorithm (FSGA)

ML framework using Genetic Algorithms for automated feature subset selection.

## Quick Start

```bash
# Setup
cd ~/code/feature-selection-via-genetic-algorithm
uv venv
uv pip install numpy scikit-learn

# Run test
uv run python tests/test_integration.py
```

## Usage

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
print(f"Best accuracy: {results['best_fitness']:.4f}")
print(f"Features selected: {results['best_chromosome'].sum()}")
```

## Current Status

**Phase 1 Complete** ✅
- Core GA engine with early stopping
- Basic operators (crossover, mutation, selection)
- ML integration (sklearn models, dataset loaders)
- AccuracyEvaluator for fitness
- Integration test passing

**Next**: See [PROJECT_PLAN.md](PROJECT_PLAN.md) for remaining tasks

## Architecture

```
fsga/
├── core/          # GA engine (genetic_algorithm, population)
├── operators/     # Crossover operators
├── mutations/     # Mutation operators
├── selectors/     # Selection strategies
├── evaluators/    # Fitness functions (ML-based)
├── ml/            # Model wrappers, CV strategies
├── datasets/      # Dataset loaders
├── analysis/      # Experiment framework
├── visualization/ # Plots
└── utils/         # Config, metrics, logging
```

## Features

- **Genetic Algorithm**: Population-based search with configurable operators
- **ML Integration**: Works with any sklearn classifier
- **Early Stopping**: Automatic convergence detection
- **Multiple Datasets**: Iris, Wine, Breast Cancer, Digits
- **Extensible**: Easy to add custom operators, evaluators, models

## Documentation

- [PROJECT_PLAN.md](PROJECT_PLAN.md) - Remaining implementation tasks
- [CLAUDE.md](CLAUDE.md) - AI collaboration context
- Module READMEs in each `fsga/*/` directory

## Test Results

**Iris Dataset** (4 features → 2 features):
- GA Accuracy: 95.56%
- Baseline (all features): 88.89%
- Improvement: +6.67%
- Selected: petal length, petal width

## License

MIT
