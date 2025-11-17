# Feature Selection via Genetic Algorithm (FSGA)

A production-ready framework for automated feature selection using Genetic Algorithms with comprehensive evaluation and visualization tools.

[![Tests](https://github.com/straightchlorine/feature-selection-via-genetic-algorithm/workflows/Tests/badge.svg)](https://github.com/straightchlorine/feature-selection-via-genetic-algorithm/actions)
[![PyPI version](https://badge.fury.io/py/fsga.svg)](https://badge.fury.io/py/fsga)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

FSGA provides a modular, extensible framework for feature selection using genetic algorithms. 

### Key Features

- **Modular Design**: Swappable operators, selectors, and evaluators
- **Multiple Operators**: 5 crossover types, 5 selection strategies, 3 fitness functions
- **Baseline Comparisons**: Built-in RFE, LASSO, Mutual Information, Chi², ANOVA
- **Statistical Rigor**: Wilcoxon, Mann-Whitney, Cohen's d, Jaccard stability
- **Visualization**: 9 publication-quality plot functions
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
| **Breast Cancer** | 98.3% | 12/30 (40%) | +2.6% vs all features |
| **Iris** | 98.3% | 2/4 (50%) | +6.2% vs all features |
| **Wine** | 100% | 6.5/13 (50%) | +1.4% vs all features |

## Installation

```bash
pip install fsga
```

Or install from source:

```bash
git clone https://github.com/straightchlorine/feature-selection-via-genetic-algorithm.git
cd feature-selection-via-genetic-algorithm
pip install -e .
```

## Documentation Structure

- **[Getting Started](getting-started/installation.md)** - Installation and quick start
- **[User Guide](user-guide/architecture.md)** - Detailed usage and architecture
- **[Project Plan](about/project-plan.md)** - Development roadmap and status

## Project Status

**Current**: Production-ready MVP with 5,161 lines of code, 280+ tests (82% coverage)

**Delivered**:
- ✅ Full-featured GA framework
- ✅ Comprehensive experiment runner
- ✅ Publication-quality visualizations
- ✅ Statistical analysis suite
- ✅ Extensive documentation

**Future Extensions**:
- Parallel fitness evaluation
- NSGA-II multi-objective optimization
- Regression support
- Deep learning integration

See [Project Plan](about/project-plan.md) for roadmap.

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

MIT License - see LICENSE file in the repository for details.
