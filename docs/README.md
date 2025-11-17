# FSGA Documentation

Complete documentation for the Feature Selection via Genetic Algorithm framework.

## Quick Navigation

### Getting Started
- **[Installation](getting-started/installation.md)** - Setup and installation instructions
- **[Quick Start](getting-started/quickstart.md)** - Get running in minutes
- **[Tutorial](getting-started/tutorial.md)** - Comprehensive step-by-step walkthrough

### Technical Documentation
- **[Architecture](user-guide/architecture.md)** - System design, patterns, and extension points
- **[Project Plan](../PROJECT_PLAN.md)** - Development roadmap and remaining tasks
- **[CLAUDE.md](../CLAUDE.md)** - AI collaboration context and project history

### Component Documentation

Module-specific documentation in the source code:

#### Core Components
- `fsga/core/README.md` - GA engine overview
- `fsga/core/README_DEV.md` - Developer guide with architecture details

#### Operators & Strategies
- `fsga/operators/README.md` - Crossover operators (5 types)
- `fsga/mutations/README.md` - Mutation operators
- `fsga/selectors/README.md` - Selection strategies (5 types)
- `fsga/evaluators/README.md` - Fitness functions (3 types)

#### ML Integration
- `fsga/ml/README.md` - Model wrappers and sklearn integration
- `fsga/datasets/README.md` - Dataset loaders

#### Analysis & Visualization
- `fsga/analysis/README.md` - Experiment framework and baselines
- `fsga/visualization/README.md` - Plotting functions (9 types)

#### Utilities
- `fsga/utils/README.md` - Configuration, metrics, and serialization

## Documentation Overview

### For Users

**New to FSGA?**
1. Start with [Installation](getting-started/installation.md)
2. Follow the [Quick Start](getting-started/quickstart.md) or [Tutorial](getting-started/tutorial.md) for hands-on examples
3. Review [Architecture](user-guide/architecture.md) to understand the system

**Running Experiments?**
- See `fsga/analysis/README.md` for ExperimentRunner
- See `fsga/visualization/README.md` for plotting
- Check example scripts in `experiments/`

### For Developers

**Contributing?**
1. Read [Architecture](user-guide/architecture.md) for system design
2. Check `fsga/core/README_DEV.md` for implementation details
3. Review module READMEs for extension points

**Adding Components?**
- New crossover: `fsga/operators/README.md`
- New selector: `fsga/selectors/README.md`
- New evaluator: `fsga/evaluators/README.md`
- New plot: `fsga/visualization/README.md`

## Key Resources

### Configuration Files
- `configs/default.yaml` - Standard settings
- `configs/quick_test.yaml` - Fast testing
- `configs/production.yaml` - Full experiments
- `configs/high_dimensional.yaml` - Large feature spaces

### Experiment Scripts
- `experiments/run_comparison.py` - GA vs baselines
- `experiments/run_comprehensive_analysis.py` - Full visualization suite

### Academic Report
- `report/` - LaTeX academic report (Polish)
- `report/README.md` - Report compilation instructions

## API Quick Reference

### Core Classes

```python
from fsga.core.genetic_algorithm import GeneticAlgorithm
from fsga.analysis.experiment_runner import ExperimentRunner
```

### Operators

```python
from fsga.operators.uniform_crossover import UniformCrossover
from fsga.mutations.bitflip_mutation import BitFlipMutation
from fsga.selectors.tournament_selector import TournamentSelector
```

### Evaluators

```python
from fsga.evaluators.accuracy_evaluator import AccuracyEvaluator
from fsga.evaluators.f1_evaluator import F1Evaluator
from fsga.evaluators.balanced_accuracy_evaluator import BalancedAccuracyEvaluator
```

### Visualization

```python
from fsga.visualization import (
    plot_fitness_evolution,
    plot_method_comparison,
    plot_multi_metric_comparison
)
```

## Statistics

- **Operators**: 5 crossover, 5 selectors, 3 evaluators
- **Baselines**: 6 comparison methods
- **Visualizations**: 9 plot functions

## Support

- **Issues**: Open an issue on GitHub
- **Questions**: See module READMEs or CLAUDE.md
- **Contributing**: Check Architecture.md for extension points
