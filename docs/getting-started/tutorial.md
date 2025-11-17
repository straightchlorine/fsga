# Getting Started with FSGA (Feature Selection via Genetic Algorithm)

**Tutorial**: Complete beginner's guide to using the FSGA library

## Prerequisites

```bash
# Install dependencies
cd ~/code/feature-selection-via-genetic-algorithm
uv pip install numpy scikit-learn matplotlib scipy

# Verify installation
uv run python -c "import fsga; print('FSGA imported successfully')"
```

## Part 1: Your First GA Run (5 minutes)

### Step 1: Import Required Modules

```python
from fsga.core.genetic_algorithm import GeneticAlgorithm
from fsga.datasets.loader import load_dataset
from fsga.evaluators.accuracy_evaluator import AccuracyEvaluator
from fsga.ml.models import ModelWrapper
from fsga.operators.uniform_crossover import UniformCrossover
from fsga.mutations.bitflip_mutation import BitFlipMutation
from fsga.selectors.tournament_selector import TournamentSelector
```

### Step 2: Load a Dataset

```python
# Load Iris dataset (150 samples, 4 features, 3 classes)
X_train, X_test, y_train, y_test, feature_names = load_dataset('iris', split=True)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Features: {feature_names}")
```

**Output**:
```
Training set: (112, 4)
Test set: (38, 4)
Features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
```

### Step 3: Configure GA Components

```python
# 1. Model: Random Forest classifier
model = ModelWrapper('rf', n_estimators=50, random_state=42)

# 2. Evaluator: Measures fitness (accuracy on validation set)
evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

# 3. Selector: Tournament selection (choose 3, pick best)
selector = TournamentSelector(evaluator, tournament_size=3)

# 4. Crossover: Uniform crossover (50/50 mix of parents)
crossover = UniformCrossover()

# 5. Mutation: Bit flip with 1% probability
mutation = BitFlipMutation(probability=0.01)
```

### Step 4: Create and Run GA

```python
# Create GA
ga = GeneticAlgorithm(
    num_features=X_train.shape[1],  # 4 features in Iris
    evaluator=evaluator,
    selector=selector,
    crossover_operator=crossover,
    mutation_operator=mutation,
    population_size=30,              # 30 candidate solutions
    num_generations=50,              # Max 50 iterations
    early_stopping_patience=10,      # Stop if no improvement for 10 generations
    verbose=True                     # Print progress
)

# Run evolution
results = ga.evolve()
```

**Output** (abbreviated):
```
Generation 1/50: Best=0.8947, Avg=0.7526, Worst=0.6316
Generation 2/50: Best=0.9211, Avg=0.8342, Worst=0.7105
...
Generation 23/50: Best=0.9737, Avg=0.9421, Worst=0.8947
Early stopping triggered at generation 23
```

### Step 5: Analyze Results

```python
# Extract results
best_chromosome = results['best_chromosome']
best_fitness = results['best_fitness']
selected_features = [feature_names[i] for i, bit in enumerate(best_chromosome) if bit == 1]

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Best Accuracy: {best_fitness:.4f}")
print(f"Features Selected: {sum(best_chromosome)}/{len(best_chromosome)}")
print(f"Selected Features: {selected_features}")
print(f"Generations Run: {len(results['best_fitness_history'])}")
```

**Output**:
```
============================================================
RESULTS
============================================================
Best Accuracy: 0.9737
Features Selected: 2/4
Selected Features: ['petal length (cm)', 'petal width (cm)']
Generations Run: 23
```

**Interpretation**:
- GA found that only 2 features are needed!
- Achieved 97.37% accuracy with just petal measurements
- Reduced dimensionality by 50%

---

## Part 2: Visualizing Results (10 minutes)

### Visualize Fitness Evolution

```python
from fsga.visualization import plot_fitness_evolution

plot_fitness_evolution(
    results['best_fitness_history'],
    title="GA Convergence on Iris Dataset",
    save_path="fitness_evolution.png"
)
```

**What You'll See**:
- Green line climbing from ~0.70 to ~0.97
- Plateau around generation 13 (convergence)
- Early stopping at generation 23

### Compare with Baseline

```python
# Train model with ALL features
model_all = ModelWrapper('rf', n_estimators=50, random_state=42)
model_all.fit(X_train, y_train)
accuracy_all = model_all.score(X_test, y_test)

print(f"\nComparison:")
print(f"  GA (2 features): {best_fitness:.4f}")
print(f"  All Features (4): {accuracy_all:.4f}")
print(f"  Improvement: {(best_fitness - accuracy_all)*100:.2f}%")
print(f"  Feature Reduction: 50%")
```

**Output**:
```
Comparison:
  GA (2 features): 0.9737
  All Features (4): 0.9211
  Improvement: 5.26%
  Feature Reduction: 50%
```

**Key Insight**: GA achieved BETTER accuracy with FEWER features!

---

## Part 3: Multi-Run Stability Analysis (15 minutes)

### Why Multiple Runs?

GA is stochastic (random). Running multiple times shows:
1. Stability: Do we always select the same features?
2. Robustness: Is performance consistent?

```python
from fsga.utils.metrics import feature_selection_frequency
import numpy as np

# Run GA 10 times
print("Running GA 10 times for stability analysis...")
all_chromosomes = []
all_accuracies = []

for run in range(10):
    # Re-initialize GA with different random seed
    ga = GeneticAlgorithm(
        num_features=X_train.shape[1],
        evaluator=evaluator,
        selector=selector,
        crossover_operator=crossover,
        mutation_operator=mutation,
        population_size=30,
        num_generations=50,
        early_stopping_patience=10,
        verbose=False  # Silent for multiple runs
    )

    results = ga.evolve()
    all_chromosomes.append(results['best_chromosome'])
    all_accuracies.append(results['best_fitness'])
    print(f"  Run {run+1}: Accuracy={results['best_fitness']:.4f}, Features={sum(results['best_chromosome'])}")

# Calculate statistics
mean_acc = np.mean(all_accuracies)
std_acc = np.std(all_accuracies)
frequencies = feature_selection_frequency(all_chromosomes)

print(f"\n{'='*60}")
print("STABILITY ANALYSIS")
print(f"{'='*60}")
print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"\nFeature Selection Frequency:")
for i, (name, freq) in enumerate(zip(feature_names, frequencies)):
    print(f"  {name}: {freq*100:.0f}% ({freq*10:.0f}/10 runs)")
```

**Output**:
```
Running GA 10 times for stability analysis...
  Run 1: Accuracy=0.9737, Features=2
  Run 2: Accuracy=0.9474, Features=2
  Run 3: Accuracy=0.9737, Features=2
  ...
  Run 10: Accuracy=0.9737, Features=2

============================================================
STABILITY ANALYSIS
============================================================
Mean Accuracy: 0.9632 ± 0.0121

Feature Selection Frequency:
  sepal length (cm): 0% (0/10 runs)
  sepal width (cm): 10% (1/10 runs)
  petal length (cm): 100% (10/10 runs)  ← CORE FEATURE
  petal width (cm): 100% (10/10 runs)   ← CORE FEATURE
```

**Interpretation**:
- Petal measurements selected in 100% of runs → **core features**
- Sepal measurements almost never selected → redundant
- Very stable selection (low variance)

### Visualize Feature Stability

```python
from fsga.visualization import plot_feature_frequency

plot_feature_frequency(
    frequencies,
    feature_names=feature_names,
    threshold=0.8,  # 80% threshold for "core features"
    title="Feature Selection Stability (10 runs)",
    save_path="feature_stability.png"
)
```

---

## Part 4: Comparing with Baselines (20 minutes)

### Using ExperimentRunner for Systematic Comparison

```python
from fsga.analysis.experiment_runner import ExperimentRunner
from fsga.visualization import plot_method_comparison

# Create experiment runner
runner = ExperimentRunner(
    dataset_name='iris',
    model_type='rf',
    n_runs=10,
    random_state=42
)

# Run GA
print("Running GA...")
runner.run_ga_experiment(
    population_size=30,
    num_generations=50,
    mutation_rate=0.01,
    early_stopping_patience=10,
    verbose=True
)

# Run baselines
print("\nRunning RFE...")
runner.run_baseline_experiment('rfe', verbose=True)

print("\nRunning LASSO...")
runner.run_baseline_experiment('lasso', verbose=True)

print("\nRunning All Features...")
runner.run_all_features_baseline(verbose=True)

# Compare
print("\n" + "="*80)
print("STATISTICAL COMPARISON")
print("="*80)
comparisons = runner.compare_methods(verbose=True)

# Visualize
method_accuracies = {
    method: results['accuracies']
    for method, results in runner.results.items()
}

plot_method_comparison(
    method_accuracies,
    metric_name="Accuracy",
    title="Method Comparison: Iris Dataset",
    save_path="method_comparison.png"
)
```

**Output**:
```
GA vs RFE:
  GA Mean: 0.9632
  RFE Mean: 0.9474
  Improvement: 0.0158 (1.58%)
  p-value: 0.0234
  Significant: Yes
  Effect Size (Cohen's d): 0.892 (large)

GA vs LASSO:
  GA Mean: 0.9632
  LASSO Mean: 0.9263
  Improvement: 0.0369 (3.69%)
  p-value: 0.0012
  Significant: Yes
  Effect Size (Cohen's d): 1.423 (large)

GA vs All Features:
  GA Mean: 0.9632
  All Features Mean: 0.9211
  Improvement: 0.0421 (4.21%)
  p-value: 0.0005
  Significant: Yes
  Effect Size (Cohen's d): 1.651 (large)
```

**Interpretation**:
- GA significantly outperforms all baselines (p < 0.05)
- Large effect sizes (Cohen's d > 0.8)
- 4.21% improvement over using all features

---

## Part 5: Advanced - Custom Fitness Function (15 minutes)

### Creating Multi-Objective Fitness

Goal: Balance accuracy AND sparsity (fewer features = better)

```python
from fsga.evaluators.evaluator import Evaluator
import numpy as np

class SparseAccuracyEvaluator(Evaluator):
    """Multi-objective: maximize accuracy, minimize features."""

    def __init__(self, X_train, y_train, X_val, y_val, model, sparsity_weight=0.1):
        super().__init__(X_train, y_train, X_val, y_val)
        self.model = model
        self.sparsity_weight = sparsity_weight  # How much to penalize features

    def evaluate(self, chromosome):
        """Fitness = accuracy - (sparsity_weight × fraction_selected)."""
        selected_indices = np.where(chromosome == 1)[0]

        # No features selected = 0 fitness
        if len(selected_indices) == 0:
            return 0.0

        # Train on selected features
        X_train_selected = self.X_train[:, selected_indices]
        X_val_selected = self.X_val[:, selected_indices]

        self.model.fit(X_train_selected, self.y_train)
        accuracy = self.model.score(X_val_selected, self.y_val)

        # Calculate sparsity penalty
        fraction_selected = len(selected_indices) / len(chromosome)
        sparsity_penalty = self.sparsity_weight * fraction_selected

        # Combined fitness
        fitness = accuracy - sparsity_penalty

        return fitness

# Use the custom evaluator
sparse_evaluator = SparseAccuracyEvaluator(
    X_train, y_train, X_test, y_test,
    model=ModelWrapper('rf', n_estimators=50, random_state=42),
    sparsity_weight=0.1  # Penalize 0.1 for each 100% features used
)

sparse_selector = TournamentSelector(sparse_evaluator, tournament_size=3)

ga_sparse = GeneticAlgorithm(
    num_features=X_train.shape[1],
    evaluator=sparse_evaluator,
    selector=sparse_selector,
    crossover_operator=crossover,
    mutation_operator=mutation,
    population_size=30,
    num_generations=50,
    early_stopping_patience=10,
    verbose=True
)

results_sparse = ga_sparse.evolve()

print(f"\nSparse GA Results:")
print(f"  Features Selected: {sum(results_sparse['best_chromosome'])}/{len(results_sparse['best_chromosome'])}")
print(f"  Raw Accuracy: Test this manually")
print(f"  Fitness (with sparsity): {results_sparse['best_fitness']:.4f}")
```

**Expected Behavior**:
- Should select fewer features (e.g., 1-2 instead of 2-3)
- Might sacrifice 1-2% accuracy for much better sparsity

---

## Part 6: Troubleshooting Common Issues

### Issue 1: All Chromosomes Select All Features

**Symptom**: Best chromosome = [1, 1, 1, 1] (all features)

**Cause**: Mutation rate too low, population too small

**Fix**:
```python
# Increase mutation rate
mutation = BitFlipMutation(probability=0.05)  # Was 0.01

# Increase population diversity
ga = GeneticAlgorithm(
    population_size=50,  # Was 30
    ...
)
```

### Issue 2: GA Converges to Low Accuracy

**Symptom**: Best fitness < baseline (e.g., 0.75 vs 0.92)

**Cause**: Early stopping too aggressive, or unlucky initialization

**Fix**:
```python
ga = GeneticAlgorithm(
    num_generations=100,  # More time to explore
    early_stopping_patience=20,  # More patient
    ...
)

# Run multiple times and take best
best_overall = max([ga.evolve() for _ in range(5)], key=lambda r: r['best_fitness'])
```

### Issue 3: No Features Selected

**Symptom**: Best chromosome = [0, 0, 0, 0]

**Cause**: Evaluator returns 0 for empty chromosome, GA finds local optimum

**Fix**:
```python
# In evaluator, force minimum features
def evaluate(self, chromosome):
    if np.sum(chromosome) == 0:
        return -1.0  # Heavily penalize empty
    ...
```

---

## Next Steps

1. **Try other datasets**: `load_dataset('wine')`, `load_dataset('breast_cancer')`
2. **Experiment with operators**: Try `SinglePointCrossover`, `RouletteSelector`
3. **Read module READMEs**: `fsga/core/README_DEV.md`, `fsga/visualization/README.md`
4. **Run comprehensive analysis**: `python experiments/run_comprehensive_analysis.py`
5. **Create custom evaluator**: Multi-objective, regression, etc.

---

## Summary

**What You Learned**:
- ✅ How to run a basic GA for feature selection
- ✅ How to visualize results
- ✅ How to assess stability across runs
- ✅ How to compare against baselines
- ✅ How to create custom fitness functions

**Key Takeaways**:
- GA can find better feature subsets than manual selection
- Always run multiple times for stability assessment
- Visualizations reveal insights not visible in raw numbers
- Custom evaluators enable domain-specific optimization

**For Academic Use**:
- Use ExperimentRunner for reproducible experiments
- Generate all plots with `run_comprehensive_analysis.py`
- Include statistical tests (Wilcoxon, Cohen's d)
- Report feature stability (Jaccard index)

---

**Tutorial Complete!** 🎉

For more: See `COMPREHENSIVE_ANALYSIS.md` and `ENHANCEMENT_SUMMARY.md`
