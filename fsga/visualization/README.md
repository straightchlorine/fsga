# fsga.visualization - Plotting & Visualization

Publication-quality visualizations for GA results and comparisons.

## Components

### `fitness_plots.py`
Fitness evolution over generations.

```python
from fsga.visualization.fitness_plots import plot_fitness_evolution

plot_fitness_evolution(
    ga_results,
    title='GA Fitness Evolution',
    save_path='results/plots/fitness.png'
)
```

**Output**: Line plot with:
- Best fitness (blue)
- Average fitness (green)
- Worst fitness (red)
- Std deviation bands (shaded)

**Options**:
```python
plot_fitness_evolution(
    results,
    show_std=True,        # Show std bands
    show_diversity=True,  # Add diversity subplot
    log_scale=False,      # Use log scale for y-axis
    smoothing=0.1         # Apply smoothing (0.0 = none, 1.0 = maximum)
)
```

### `pareto_plots.py`
Pareto frontier for multi-objective optimization.

```python
from fsga.visualization.pareto_plots import plot_pareto_frontier

# For multi-objective results
plot_pareto_frontier(
    pareto_solutions,
    objective1_name='Accuracy',
    objective2_name='Feature Sparsity',
    highlight_solutions=[0, 5, 10],  # Highlight specific solutions
    save_path='results/plots/pareto.png'
)
```

**Output**: Scatter plot showing:
- Pareto-optimal solutions (blue, larger markers)
- Dominated solutions (gray, small markers)
- Trade-off curve

**Use case**: Show decision-maker the accuracy vs. sparsity trade-off

### `feature_importance.py`
Feature selection heatmaps and stability plots.

```python
from fsga.visualization.feature_importance import plot_feature_stability

# Run GA multiple times
all_chromosomes = [run_ga() for _ in range(20)]

plot_feature_stability(
    chromosomes=all_chromosomes,
    feature_names=feature_names,
    save_path='results/plots/stability_heatmap.png'
)
```

**Output**: Heatmap showing how often each feature is selected across runs
- Dark green: always selected (100%)
- Light green: sometimes selected (50%)
- White: never selected (0%)

**Interpretation**: Stable features = robust predictors

### `convergence_plots.py`
Convergence analysis visualizations.

```python
from fsga.visualization.convergence_plots import (
    plot_convergence_analysis,
    plot_diversity_vs_fitness
)

# Overall convergence
plot_convergence_analysis(
    results,
    save_path='results/plots/convergence.png'
)
# Shows: fitness curve + diversity + convergence detection

# Diversity-fitness relationship
plot_diversity_vs_fitness(results)
# Scatter: does higher diversity correlate with better fitness?
```

### `comparison_plots.py`
Compare GA against baseline methods.

```python
from fsga.visualization.comparison_plots import (
    plot_method_comparison,
    plot_accuracy_vs_features
)

# Bar chart comparison
plot_method_comparison(
    comparison_results,
    metric='accuracy',
    save_path='results/plots/method_comparison.png'
)

# Accuracy vs. # features scatter
plot_accuracy_vs_features(
    comparison_results,
    highlight_method='GA',
    save_path='results/plots/accuracy_vs_features.png'
)
```

**Output**:
- Bar chart: accuracy for each method (with error bars)
- Scatter: shows if GA achieves better accuracy with fewer features

## Advanced Visualizations

### Multi-Run Comparison
```python
from fsga.visualization.comparison_plots import plot_multi_run_comparison

# Results from 30 runs of each method
plot_multi_run_comparison(
    {
        'GA': ga_accuracies,
        'RFE': rfe_accuracies,
        'LASSO': lasso_accuracies
    },
    plot_type='violin',  # or 'box', 'swarm'
    save_path='results/plots/multi_run.png'
)
```

**Output**: Violin plot showing distribution of accuracies (not just mean)

### Statistical Significance Annotations
```python
from fsga.visualization.comparison_plots import plot_with_significance

plot_with_significance(
    method_results,
    baseline='All Features',
    p_values={'GA': 0.003, 'RFE': 0.042, 'LASSO': 0.231},
    save_path='results/plots/significance.png'
)
# Adds * (p<0.05), ** (p<0.01), *** (p<0.001) above bars
```

### Feature Selection Timeline
```python
from fsga.visualization.fitness_plots import plot_feature_selection_timeline

plot_feature_selection_timeline(
    ga_results,
    feature_names=feature_names,
    save_path='results/plots/timeline.png'
)
```

**Output**: Heatmap over generations showing which features are in the best chromosome
- Shows when features are added/removed
- Identifies features that persist vs. fluctuate

## Customization

### Style Configuration
```python
from fsga.visualization.style import set_publication_style

set_publication_style(
    font_size=12,
    figure_size=(10, 6),
    dpi=300,  # High resolution for papers
    style='seaborn-v0_8-paper'  # or 'ggplot', 'bmh'
)

# Now all plots use this style
plot_fitness_evolution(results)
```

### Color Schemes
```python
from fsga.visualization.colors import ColorScheme

colors = ColorScheme.get_palette('colorblind_safe')
# Returns list of colors that work for colorblind readers

plot_method_comparison(
    results,
    colors=colors
)
```

## Export Formats

### Save Options
```python
# PNG (for presentations)
plot_fitness_evolution(results, save_path='fitness.png', dpi=150)

# PDF (for papers, vector graphics)
plot_fitness_evolution(results, save_path='fitness.pdf')

# SVG (for editing in Inkscape/Illustrator)
plot_fitness_evolution(results, save_path='fitness.svg')

# Multiple formats
plot_fitness_evolution(results, save_path='fitness', formats=['png', 'pdf', 'svg'])
```

## Interactive Plots (Optional)

### Plotly Integration
```python
from fsga.visualization.interactive import plot_interactive_pareto

# Generates interactive HTML plot
plot_interactive_pareto(
    pareto_solutions,
    save_path='results/plots/pareto_interactive.html'
)
# Hover to see exact values, zoom, pan
```

### Jupyter Integration
```python
# In Jupyter notebook
%matplotlib inline
from fsga.visualization.fitness_plots import plot_fitness_evolution

plot_fitness_evolution(results)  # Shows inline
```

## Example Workflow

```python
from fsga.visualization import (
    plot_fitness_evolution,
    plot_feature_stability,
    plot_method_comparison
)

# After running GA
plot_fitness_evolution(ga_results, save_path='results/plots/1_fitness.png')

# After multiple runs
plot_feature_stability(all_chromosomes, feature_names, save_path='results/plots/2_stability.png')

# After comparison with baselines
plot_method_comparison(comparison_results, save_path='results/plots/3_comparison.png')

# All plots saved to results/plots/ for inclusion in report
```

## Best Practices

1. **High DPI for papers** (300 dpi), lower for web (150 dpi)
2. **Meaningful titles** (avoid generic "Plot 1")
3. **Clear legends** (spell out acronyms)
4. **Error bars** (always show variability)
5. **Consistent colors** (same method = same color across plots)
6. **Accessibility** (use colorblind-safe palettes)

## Extending

Add custom plot:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_my_custom_viz(data, save_path=None):
    """Your custom visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Your plotting code
    ax.plot(data['x'], data['y'])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_title('My Custom Plot')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
```
