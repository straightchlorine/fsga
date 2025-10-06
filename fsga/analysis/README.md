# fsga.analysis - Experimental Analysis

Tools for running experiments, comparing methods, and statistical testing.

## Components

### `experiment_runner.py`
Orchestrate GA experiments with multiple configurations.

```python
from fsga.analysis.experiment_runner import ExperimentRunner

runner = ExperimentRunner(
    datasets=['iris', 'wine', 'breast_cancer'],
    models=['rf', 'svm', 'logistic'],
    selectors=[TournamentSelector, RouletteSelector],
    n_runs=10  # Repeat each config 10 times
)

results = runner.run_all()
# Returns DataFrame with all results
```

**Output**:
```
| dataset | model | selector | run | accuracy | n_features | time_ms |
|---------|-------|----------|-----|----------|------------|---------|
| iris    | rf    | Tournament| 1   | 0.9667   | 2          | 145.3   |
| iris    | rf    | Tournament| 2   | 0.9333   | 3          | 132.1   |
...
```

### `comparison.py`
Compare GA against baseline feature selection methods.

```python
from fsga.analysis.comparison import FeatureSelectionComparison

comparison = FeatureSelectionComparison(X_train, y_train, X_test, y_test)

# Run all methods
results = comparison.run_all(
    ga_config={...},
    rfe_params={'n_features_to_select': 10},
    lasso_params={'alpha': 0.01}
)

# Get comparison table
comparison.print_summary()
```

**Baseline Methods**:
1. **RFE** (Recursive Feature Elimination): sklearn wrapper method
2. **LASSO** (L1 regularization): embedded method
3. **Mutual Information**: filter method
4. **Chi-Squared**: filter method (for classification)
5. **Random Forest Importance**: embedded method
6. **All Features**: no selection (upper bound)

**Output**:
```
Method              | Accuracy | Features | Time (ms) | Stability
--------------------|----------|----------|-----------|----------
GA (Tournament)     | 0.9423   | 6/13     | 2341.2    | 0.82
RFE                 | 0.9312   | 8/13     | 1823.4    | 0.91
LASSO (α=0.01)      | 0.9187   | 5/13     | 145.7     | 0.76
Mutual Information  | 0.9021   | 6/13     | 23.4      | 1.00
All Features        | 0.9134   | 13/13    | 87.3      | 1.00
```

### `statistical_tests.py`
Determine if GA is statistically significantly better.

```python
from fsga.analysis.statistical_tests import compare_methods

# Run GA 30 times
ga_accuracies = [run_ga() for _ in range(30)]

# Run RFE 30 times
rfe_accuracies = [run_rfe() for _ in range(30)]

# Wilcoxon signed-rank test (paired)
result = compare_methods(ga_accuracies, rfe_accuracies, test='wilcoxon')

print(f"p-value: {result['p_value']}")
print(f"Significant: {result['significant']}")  # True if p < 0.05
print(f"Effect size: {result['effect_size']}")  # Cohen's d
```

**Available Tests**:
- `wilcoxon`: Paired, non-parametric (GA run i vs RFE run i)
- `mann_whitney`: Unpaired, non-parametric
- `t_test`: Paired, parametric (assumes normality)
- `friedman`: Multiple methods comparison

### `convergence_analyzer.py`
Analyze GA convergence behavior.

```python
from fsga.analysis.convergence_analyzer import ConvergenceAnalyzer

analyzer = ConvergenceAnalyzer(ga_results)

# Detect convergence point
conv_gen = analyzer.detect_convergence(
    patience=10,  # No improvement for 10 generations
    threshold=0.001  # < 0.1% fitness change
)
print(f"Converged at generation {conv_gen}")

# Analyze diversity loss
analyzer.plot_diversity_vs_fitness()
# Shows if premature convergence occurred

# Early stopping recommendation
recommended_gens = analyzer.recommend_max_generations()
print(f"Recommend max_generations={recommended_gens}")
```

## Usage Examples

### Example 1: Full Comparison Study
```python
from fsga.analysis.comparison import FeatureSelectionComparison
from fsga.datasets.loader import load_dataset

# Load data
X_train, X_test, y_train, y_test, names = load_dataset('wine', split=True)

# Run comparison
comparison = FeatureSelectionComparison(X_train, y_train, X_test, y_test)
results = comparison.run_all()

# Statistical tests
comparison.statistical_comparison(baseline='RFE')
# Automatically runs Wilcoxon test vs each method

# Save results
comparison.save_results('results/wine_comparison.csv')
```

### Example 2: Parameter Sensitivity Analysis
```python
from fsga.analysis.experiment_runner import ExperimentRunner

# Test different population sizes
runner = ExperimentRunner()
results = runner.grid_search({
    'population_size': [10, 25, 50, 100],
    'mutation_rate': [0.01, 0.05, 0.1],
    'selector': ['tournament', 'roulette']
})

# Best configuration
best = results.loc[results['accuracy'].idxmax()]
print(f"Best config: {best}")
```

### Example 3: Stability Analysis
```python
from fsga.analysis.stability import FeatureStabilityAnalyzer

# Run GA 20 times
selected_features = []
for _ in range(20):
    ga = GeneticAlgorithm(...)
    results = ga.evolve()
    selected_features.append(results['best_chromosome'])

# Analyze stability
analyzer = FeatureStabilityAnalyzer(selected_features, feature_names)
stability_score = analyzer.jaccard_stability()
print(f"Stability: {stability_score:.2f}")  # 1.0 = always selects same features

# Which features are consistently selected?
core_features = analyzer.get_core_features(threshold=0.8)
print(f"Core features (>80% selection): {core_features}")
```

## Metrics Explained

### Stability
How consistent is feature selection across multiple runs?

**Jaccard Index**:
```
J(A, B) = |A ∩ B| / |A ∪ B|
```

- 1.0 = always selects exactly the same features
- 0.5 = 50% overlap on average
- 0.0 = completely random

**Why it matters**: High accuracy with low stability → not trustworthy

### Effect Size (Cohen's d)
How large is the performance difference?

```
d = (mean_GA - mean_baseline) / pooled_std
```

- d > 0.8: Large effect
- d > 0.5: Medium effect
- d > 0.2: Small effect
- d < 0.2: Negligible

**Interpretation**: p < 0.05 means "statistically significant", but effect size tells you if it's **practically significant**

## Best Practices

1. **Run multiple times** (≥10, ideally 30) for robust statistics
2. **Use same train/test split** across methods (fairness)
3. **Report both mean and std** (not just mean)
4. **Check assumptions** (normality for t-test, use Wilcoxon if violated)
5. **Bonferroni correction** for multiple comparisons (p_threshold = 0.05/n_tests)

## Extending

Add new baseline method:

```python
# In comparison.py
def _run_my_method(self, X_train, y_train, X_test, y_test, **kwargs):
    # Your feature selection logic
    selected_features = your_selection_algorithm(X_train, y_train)

    # Evaluate
    X_train_subset = X_train[:, selected_features]
    X_test_subset = X_test[:, selected_features]

    model = self.model.fit(X_train_subset, y_train)
    accuracy = model.score(X_test_subset, y_test)

    return {
        'accuracy': accuracy,
        'n_features': len(selected_features),
        'selected_features': selected_features
    }
```
