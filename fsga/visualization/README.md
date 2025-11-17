# Visualization Module

**Purpose**: Publication-quality plotting functions for GA analysis, method comparison, and result visualization.

## Overview

This module provides 9 comprehensive plotting functions designed for academic publication and technical analysis. All plots support 300 DPI export, consistent styling, and optional display control.

## Core Design Principles

1. **Publication-Ready**: 300 DPI PNG export, clean typography, professional styling
2. **Comprehensive**: Cover all aspects of GA behavior (fitness, diversity, features, comparison)
3. **Flexible**: Optional parameters for customization
4. **Consistent**: Unified color schemes and formatting across all plots

## Plot Functions

### 1. Single-Run Diagnostics

#### `plot_fitness_evolution()`
Shows best/average/worst fitness over generations.

**Use Case**: Verify convergence, detect plateaus, assess optimization progress.

```python
from fsga.visualization import plot_fitness_evolution

results = ga.evolve()
plot_fitness_evolution(
    results['best_fitness_history'],
    results.get('avg_fitness_history'),
    title="GA Convergence on Iris",
    save_path="fitness_evolution.png"
)
```

**What It Shows**:
- Green line: Best fitness trajectory
- Blue line: Average fitness (population quality)
- Red line: Worst fitness (diversity indicator)

**Insights**:
- Steep initial climb → Good exploration
- Plateau → Convergence or premature convergence
- Gap between best/avg → Population diversity

---

#### `plot_diversity_evolution()`
Shows population diversity (genetic variety) over generations.

**Use Case**: Detect premature convergence, verify exploration-exploitation balance.

```python
from fsga.visualization import plot_diversity_evolution
from fsga.utils.metrics import population_diversity

diversity_history = [
    population_diversity(pop.chromosomes)
    for pop in ga.population_history
]

plot_diversity_evolution(
    diversity_history,
    title="Population Diversity",
    save_path="diversity.png"
)
```

**What It Shows**:
- Purple line: Diversity metric (0-1 scale)
- Red dashed line: Low diversity threshold (0.1)

**Insights**:
- High diversity early → Good exploration
- Gradual decline → Normal convergence
- Rapid drop → Premature convergence (problematic)
- Stays above 0.1 → Healthy population

---

#### `plot_convergence()`
Highlights the generation where convergence occurred.

**Use Case**: Identify early stopping point, visualize patience window.

```python
from fsga.visualization import plot_convergence
from fsga.utils.metrics import convergence_detected

history = results['best_fitness_history']
conv_gen = convergence_detected(history, patience=10)

plot_convergence(
    history,
    convergence_gen=conv_gen,
    patience=10,
    save_path="convergence.png"
)
```

**What It Shows**:
- Green line: Fitness evolution
- Red vertical line: Convergence point
- Yellow shaded region: Patience window
- Red dot: Final fitness at convergence

**Insights**:
- Early convergence (gen < 30) → Easy problem or lucky initialization
- Late convergence (gen > 100) → Complex problem or slow learning
- No convergence → Increase max_generations

---

### 2. Multi-Run Analysis

#### `plot_feature_frequency()`
Shows how often each feature was selected across multiple runs.

**Use Case**: Identify core features, assess feature stability, domain insights.

```python
from fsga.visualization import plot_feature_frequency
from fsga.utils.metrics import feature_selection_frequency

# Run GA 20 times
chromosomes = [ga.evolve()['best_chromosome'] for _ in range(20)]
frequencies = feature_selection_frequency(chromosomes)

plot_feature_frequency(
    frequencies,
    feature_names=dataset.feature_names,
    threshold=0.8,
    save_path="feature_freq.png"
)
```

**What It Shows**:
- Green bars: Features selected ≥80% of runs (core features)
- Gray bars: Features selected <80% of runs (peripheral)
- Red dashed line: Core feature threshold

**Insights**:
- Features at 100% → Essential for accuracy
- Features at 50-80% → Moderately important
- Features at <20% → Likely redundant or noisy

---

#### `plot_combined_dashboard()`
3-panel comprehensive view: fitness + diversity + feature frequency.

**Use Case**: Single-figure summary of GA behavior for papers/presentations.

```python
from fsga.visualization import plot_combined_dashboard

plot_combined_dashboard(
    results['best_fitness_history'],
    diversity_history,
    frequencies,
    feature_names=feature_names,
    title=f"GA Dashboard: {dataset_name}",
    save_path="dashboard.png"
)
```

**Layout**:
```
┌─────────────────┬─────────────────┐
│ Fitness Evol    │ Diversity Evol  │
├─────────────────┴─────────────────┤
│ Feature Selection Frequency       │
└───────────────────────────────────┘
```

---

### 3. Method Comparison

#### `plot_method_comparison()`
Box plots comparing accuracy across methods.

**Use Case**: Show GA vs baselines, statistical distribution, outliers.

```python
from fsga.visualization import plot_method_comparison

method_results = {
    'GA': ga_accuracies,
    'RFE': rfe_accuracies,
    'LASSO': lasso_accuracies,
    'All Features': all_feat_accuracies
}

plot_method_comparison(
    method_results,
    metric_name="Accuracy",
    title="Method Comparison: Breast Cancer",
    save_path="comparison.png"
)
```

**What It Shows**:
- Box: Quartiles (25th, median, 75th percentile)
- Green triangle: Mean
- Whiskers: Min/max within 1.5×IQR
- Circles: Outliers

---

#### `plot_feature_count_comparison()` ⭐ NEW
Box plots comparing number of features selected.

**Use Case**: Show feature reduction, sparsity comparison.

```python
from fsga.visualization import plot_feature_count_comparison

results = {
    'GA': {'n_features': np.array([12, 13, 11, 12])},
    'RFE': {'n_features': np.array([15, 15, 15, 15])},
    'All Features': {'n_features': 30}
}

plot_feature_count_comparison(
    results,
    title="Feature Count Comparison",
    save_path="feature_counts.png"
)
```

**Critical**: This plot shows the MAIN value proposition of feature selection!

---

#### `plot_multi_metric_comparison()` ⭐ NEW
2×2 grid: Accuracy + Feature Count + Runtime + Stability.

**Use Case**: Comprehensive method comparison in single figure.

```python
from fsga.visualization import plot_multi_metric_comparison

plot_multi_metric_comparison(
    runner.results,  # From ExperimentRunner
    title="Comprehensive Comparison: Iris",
    save_path="multi_metric.png"
)
```

**Layout**:
```
┌────────────────┬────────────────┐
│ Accuracy       │ Feature Count  │
├────────────────┼────────────────┤
│ Runtime (sec)  │ Stability      │
└────────────────┴────────────────┘
```

**Insights**:
- Top-left: Classification performance
- Top-right: Sparsity/dimensionality reduction
- Bottom-left: Computational efficiency
- Bottom-right: Feature selection consistency

---

#### `plot_accuracy_vs_sparsity()` ⭐ NEW
Scatter plot showing accuracy-sparsity trade-off.

**Use Case**: Visualize Pareto frontier, show GA finds optimal balance.

```python
from fsga.visualization import plot_accuracy_vs_sparsity

plot_accuracy_vs_sparsity(
    runner.results,
    total_features=30,
    title="Accuracy vs. Sparsity: Breast Cancer",
    save_path="pareto.png"
)
```

**What It Shows**:
- X-axis: Sparsity (1 - #features/#total)
- Y-axis: Accuracy
- Points: Individual runs
- Diamonds: Method means
- Goal annotation: Top-right corner

**Insights**:
- Top-right = Best (high accuracy, few features)
- Bottom-left = Worst (low accuracy, many features)
- GA typically achieves top-right position

---

## Usage Patterns

### Pattern 1: Single Experiment Visualization
```python
# Run experiment
results = ga.evolve()

# Visualize
plot_fitness_evolution(results['best_fitness_history'], save_path="fitness.png")
plot_feature_frequency(results['best_chromosome'], save_path="features.png")
```

### Pattern 2: Multi-Run Analysis
```python
# Multiple runs
runner = ExperimentRunner('iris', n_runs=20)
runner.run_ga_experiment()

# Dashboard
plot_combined_dashboard(
    runner.results['GA']['fitness_histories'][0],
    diversity_history,
    frequencies,
    save_path="dashboard.png"
)
```

### Pattern 3: Method Comparison
```python
# Run comparisons
runner.run_ga_experiment()
runner.run_baseline_experiment('rfe')
runner.run_baseline_experiment('lasso')
runner.run_all_features_baseline()

# Multi-metric comparison
plot_multi_metric_comparison(
    runner.results,
    save_path="comparison.png"
)

# Accuracy vs sparsity
plot_accuracy_vs_sparsity(
    runner.results,
    total_features=30,
    save_path="pareto.png"
)
```

## File Organization

```
fsga/visualization/
├── __init__.py          # Exports all plot functions
├── plots.py             # Implementation (706 lines)
└── README.md            # This file
```

## Plot Function Summary

| Function | Type | Key Insight | Priority |
|----------|------|-------------|----------|
| `plot_fitness_evolution()` | Diagnostics | Convergence behavior | High |
| `plot_diversity_evolution()` | Diagnostics | Exploration/exploitation | Medium |
| `plot_convergence()` | Diagnostics | Early stopping validation | Low |
| `plot_feature_frequency()` | Analysis | Core feature identification | High |
| `plot_combined_dashboard()` | Summary | Complete single-run view | High |
| `plot_method_comparison()` | Comparison | Accuracy comparison | Critical |
| `plot_feature_count_comparison()` | Comparison | Feature reduction | **Critical** |
| `plot_multi_metric_comparison()` | Comparison | All metrics at once | **Critical** |
| `plot_accuracy_vs_sparsity()` | Comparison | Trade-off visualization | High |

## Best Practices

### For Academic Papers
1. Use `plot_multi_metric_comparison()` for main results (shows all dimensions)
2. Use `plot_accuracy_vs_sparsity()` to show Pareto optimality
3. Use `plot_feature_frequency()` for domain insights (which features matter)
4. All plots at 300 DPI for publication quality

### For Presentations
1. Use `plot_combined_dashboard()` for executive summary
2. Use `plot_method_comparison()` for simple accuracy comparison
3. Use `plot_fitness_evolution()` to explain GA optimization

### For Technical Reports
1. Include all 9 plot types for comprehensive documentation
2. Use `show=False, save_path="..."` to batch-generate figures
3. Organize in subdirectories: `results/{dataset}/plots/`

## Common Issues

### Issue: Box plot looks like a line
**Cause**: Only 1 run or no variance
**Fix**: Increase `n_runs` to 10+ in ExperimentRunner

### Issue: Feature names overlapping
**Cause**: Too many features for x-axis
**Fix**: Use feature indices (`feature_names=None`) or increase figure width

### Issue: All Features baseline has no box
**Cause**: Scalar n_features (constant across runs)
**Fix**: This is expected; box will be flat line

## Extension Points

To add new plot types:

1. **Add function to `plots.py`**:
```python
def plot_my_metric(data, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    # ... plotting logic
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
```

2. **Export in `__init__.py`**:
```python
from fsga.visualization.plots import plot_my_metric
__all__ = [..., "plot_my_metric"]
```

3. **Document in README**:
Add to appropriate section with usage example.

## Dependencies

- **matplotlib**: Plotting backend
- **numpy**: Data manipulation
- **Optional**: seaborn (for enhanced styling, currently not used)

## Performance Notes

- All plots use vector graphics (300 DPI PNG)
- Large datasets (>50 features) may need wider figures
- `show=False` significantly faster for batch generation
- Dashboard plots take ~0.5s to render

## See Also

- `fsga/utils/metrics.py`: Metric calculation functions
- `fsga/analysis/experiment_runner.py`: Batch experiment framework
- `experiments/run_comparison.py`: Example usage
