# Experiment Scripts

## Main Script

**`run_experiment.py`** - Unified experiment runner with full functionality

### Usage

```bash
# Full analysis (default)
python run_experiment.py

# Quick test mode
python run_experiment.py --quick

# Specific datasets
python run_experiment.py --datasets iris wine

# Custom number of runs
python run_experiment.py --runs 20

# Skip visualizations (faster)
python run_experiment.py --no-plots
```

### Features

- Runs GA and baseline methods (RFE, LASSO, Mutual Information)
- Statistical comparison with Wilcoxon/Mann-Whitney tests
- Generates comprehensive visualization suite:
  - Method comparison (accuracy)
  - Feature count comparison
  - Multi-metric dashboard
  - Accuracy vs sparsity trade-off
  - GA fitness evolution
  - GA combined dashboard
- Saves results to `results/{mode}/{dataset}/`
- Proper logging to console and file

### Configuration

**Full mode** (default):
- Datasets: iris, wine, breast_cancer
- Runs: 10 per method
- Generations: 100

**Quick mode** (`--quick`):
- Datasets: iris only
- Runs: 3 per method
- Generations: 30

## Old Scripts (Deprecated)

- `run_comparison.py.old` - Basic comparison (deprecated, use `run_experiment.py`)
- `run_comprehensive_analysis.py.old` - With plots (deprecated, use `run_experiment.py`)

These scripts have been consolidated into `run_experiment.py` for clarity and maintainability.
