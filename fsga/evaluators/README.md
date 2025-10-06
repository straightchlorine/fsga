# fsga.evaluators - Fitness Functions

Evaluate chromosomes by training ML models on selected features.

## Components

### `evaluator.py`
Abstract base class defining the fitness interface.

**Key Difference from Knapsack**: Fitness = ML model performance (not item value)

```python
class Evaluator(ABC):
    def __init__(self, X_train, y_train, X_val, y_val, model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = model

    @abstractmethod
    def evaluate(self, chromosome: np.ndarray) -> float:
        """
        1. Select features where chromosome[i] == 1
        2. Train model on selected features
        3. Return validation performance
        """
        pass
```

### Implemented Evaluators

#### `accuracy_evaluator.py`
Simple classification accuracy.

```python
evaluator = AccuracyEvaluator(X_train, y_train, X_val, y_val, model)
fitness = evaluator.evaluate(chromosome)
# Returns: validation accuracy (0.0 to 1.0)
```

**Use case**: Balanced datasets, general purpose

#### `balanced_evaluator.py`
Balanced accuracy (average of per-class recalls).

**Use case**: Imbalanced datasets (e.g., 90% class 0, 10% class 1)

```python
from sklearn.metrics import balanced_accuracy_score
# Prevents model from just predicting majority class
```

#### `f1_evaluator.py`
F1-score (harmonic mean of precision and recall).

**Use case**: Imbalanced datasets, when false positives and false negatives have different costs

#### `multi_objective_evaluator.py` ✨
Returns tuple: (accuracy, feature_sparsity)

**For use with NSGA-II selector**

```python
evaluator = MultiObjectiveEvaluator(X_train, y_train, X_val, y_val, model)
fitness = evaluator.evaluate(chromosome)
# Returns: (0.92, 0.7)  → 92% accuracy, 70% sparsity (30% features used)
```

#### `custom_metric_evaluator.py`
User-defined scoring function.

```python
def custom_scorer(y_true, y_pred):
    # Your metric (e.g., weighted F1, custom business metric)
    return score

evaluator = CustomMetricEvaluator(
    X_train, y_train, X_val, y_val, model,
    scoring_func=custom_scorer
)
```

## Performance Considerations

### Bottleneck: Model Training

Evaluating one chromosome requires:
1. Feature selection: O(n) - fast
2. Model training: O(n × m × log(m)) - SLOW (for RF)
3. Prediction: O(n × m) - fast

**Optimization Strategies**:

1. **Caching**:
```python
# Hash chromosome → fitness mapping
cache = {}
chrom_hash = tuple(chromosome)
if chrom_hash in cache:
    return cache[chrom_hash]
```

2. **Parallel Evaluation**:
```python
from multiprocessing import Pool

with Pool(processes=4) as pool:
    fitnesses = pool.map(evaluator.evaluate, population)
```

3. **Fast Models**:
- RandomForest (fast, parallel)
- Logistic Regression (very fast)
- Avoid: SVM with RBF kernel (slow on large datasets)

## Cross-Validation Support

```python
from fsga.ml.cv_strategy import KFoldCV

evaluator = AccuracyEvaluator(
    X_train, y_train, X_val, y_val, model,
    cv_strategy=KFoldCV(n_splits=5)
)
# Fitness = average accuracy across 5 folds (more robust, but 5x slower)
```

## Extending

```python
from fsga.evaluators.evaluator import Evaluator

class MyEvaluator(Evaluator):
    def evaluate(self, chromosome):
        # 1. Select features
        selected = np.where(chromosome == 1)[0]
        if len(selected) == 0:
            return 0.0  # Penalty for no features

        # 2. Train model
        X_train_subset = self.X_train[:, selected]
        X_val_subset = self.X_val[:, selected]
        self.model.fit(X_train_subset, self.y_train)

        # 3. Evaluate
        y_pred = self.model.predict(X_val_subset)
        return your_custom_metric(self.y_val, y_pred)
```
