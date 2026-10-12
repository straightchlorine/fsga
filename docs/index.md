---
title: Feature Selection via Genetic Algorithm in Python
description: FSGA is a Python library for feature selection with genetic algorithms, with swappable operators, baselines, statistical tests, plots and an experiment runner.
---

# Feature Selection via Genetic Algorithm (FSGA)

A university project implementing feature selection using Genetic Algorithms, with evaluation and visualization tools.

[![Tests](https://github.com/straightchlorine/fsga/workflows/Tests/badge.svg)](https://github.com/straightchlorine/fsga/actions)
[![PyPI version](https://badge.fury.io/py/fsga.svg)](https://pypi.org/project/fsga/)
[![Total Downloads](https://static.pepy.tech/badge/fsga)](https://pepy.tech/project/fsga)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/fsga)](https://pypi.org/project/fsga/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

FSGA is a modular library for feature selection using genetic algorithms.

### Key Features

- **Modular Design**: Swappable operators, selectors, and evaluators
- **Multiple Operators**: 4 crossover operators, 4 selection strategies, 3 fitness functions
- **Baseline Comparisons**: Built-in RFE, LASSO, Mutual Information, Chi², ANOVA
- **Statistical Testing**: Wilcoxon, Mann-Whitney, Cohen's d, Jaccard stability
- **Visualization**: 9 plot functions for analysis and comparison
- **Experiment Framework**: `ExperimentRunner` for reproducible experiments
- **Configuration**: YAML-based configuration system

### Quick Example

```python
from fsga.core.genetic_algorithm import GeneticAlgorithm
from fsga.datasets.loader import load_dataset
from fsga.evaluators.accuracy_evaluator import AccuracyEvaluator
from fsga.ml.models import ModelWrapper

# Load data
X_train, X_test, y_train, y_test, _ = load_dataset('iris', split=True)

# Setup
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
    num_generations=100
)

results = ga.evolve()
print(f"Accuracy: {results['best_fitness']:.2%}")
print(f"Features: {results['best_chromosome'].sum()}/{X_train.shape[1]}")
```

### Results

| Dataset | GA Accuracy | Features Used | Improvement |
|---------|-------------|---------------|-------------|
| **Breast Cancer** | 98.3% | 14.3/30 (48%) | +2.6% vs all features |
| **Iris** | 97.7% | 2.6/4 (65%) | +2.7% vs all features |
| **Wine** | 100% | 7.2/13 (55%) | +1.7% vs all features |

## Installation

```bash
pip install fsga
```

Or install from source:

```bash
git clone https://github.com/straightchlorine/fsga.git
cd fsga
pip install -e .
```

## Documentation Structure

- **[Getting Started](getting-started/quickstart.md)** - Installation and quick start
- **[User Guide](user-guide/architecture.md)** - Detailed usage and architecture
- **[Project Plan](about/project-plan.md)** - Development roadmap and status

## Project Status

**Delivered**:
- GA framework with swappable operators
- Experiment runner with baseline comparisons
- Visualization plots
- Statistical tests
- Documentation

**Future Extensions**:
- Parallel fitness evaluation
- NSGA-II multi-objective optimization
- Regression support
- Deep learning integration

See [Project Plan](about/project-plan.md) for roadmap.

## License

MIT License (see `pyproject.toml`). A LICENSE file has not been added to the repository yet.
