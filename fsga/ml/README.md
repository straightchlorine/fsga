# fsga.ml - Machine Learning Integration

Wrappers and utilities for ML models, cross-validation, and preprocessing.

## Components

### `models.py`
Unified interface for scikit-learn classifiers.

```python
from fsga.ml.models import ModelWrapper

# Supported models
model = ModelWrapper('rf', n_estimators=100, random_state=42)
model = ModelWrapper('svm', kernel='rbf', C=1.0)
model = ModelWrapper('logistic', max_iter=1000)
model = ModelWrapper('knn', n_neighbors=5)
model = ModelWrapper('xgboost', n_estimators=100)
model = ModelWrapper('mlp', hidden_layer_sizes=(100,))

# Use like any sklearn model
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
```

**Supported Models**:
- `rf`: RandomForestClassifier
- `svm`: SVC
- `logistic`: LogisticRegression
- `knn`: KNeighborsClassifier
- `xgboost`: XGBClassifier (requires xgboost)
- `mlp`: MLPClassifier (neural network)

### `cv_strategy.py`
Cross-validation strategies for robust fitness evaluation.

```python
from fsga.ml.cv_strategy import KFoldCV, StratifiedKFoldCV

# K-Fold (for balanced datasets)
cv = KFoldCV(n_splits=5, shuffle=True, random_state=42)
scores = cv.evaluate(model, X, y)
mean_score = scores.mean()

# Stratified K-Fold (for imbalanced datasets)
cv = StratifiedKFoldCV(n_splits=5)
# Preserves class distribution in each fold
```

**Available Strategies**:
- `KFoldCV`: Standard k-fold
- `StratifiedKFoldCV`: Maintains class ratios
- `TimeSeriesCV`: For temporal data (no shuffling)
- `LeaveOneOutCV`: For very small datasets

### `preprocessor.py`
Data preprocessing pipeline.

```python
from fsga.ml.preprocessor import Preprocessor

preprocessor = Preprocessor(
    scale=True,              # StandardScaler
    handle_missing='mean',   # Impute with mean
    encode_categorical=True  # One-hot encoding
)

X_train, X_test = preprocessor.fit_transform(X_train, X_test)
```

**Features**:
- Scaling: StandardScaler, MinMaxScaler, RobustScaler
- Missing values: mean, median, mode, drop
- Categorical encoding: one-hot, label encoding
- Feature engineering: polynomial features, interactions

### `feature_analyzer.py`
Feature correlation and importance analysis.

```python
from fsga.ml.feature_analyzer import FeatureAnalyzer

analyzer = FeatureAnalyzer(X, y, feature_names)

# Correlation matrix
corr_matrix = analyzer.correlation_matrix()
# Returns: (num_features, num_features) correlation matrix

# Mutual information scores
mi_scores = analyzer.mutual_information()
# Returns: (num_features,) array of MI scores with target

# Feature importance (using Random Forest)
importance = analyzer.tree_importance(model='rf')

# Visualize
analyzer.plot_correlation_heatmap(save_path='corr.png')
```

## Usage Example

```python
from fsga.ml.models import ModelWrapper
from fsga.ml.cv_strategy import StratifiedKFoldCV
from fsga.ml.preprocessor import Preprocessor
from fsga.evaluators.accuracy_evaluator import AccuracyEvaluator

# 1. Preprocess
preprocessor = Preprocessor(scale=True)
X_train, X_val = preprocessor.fit_transform(X_train, X_val)

# 2. Choose model
model = ModelWrapper('rf', n_estimators=100)

# 3. Setup CV
cv = StratifiedKFoldCV(n_splits=5)

# 4. Create evaluator
evaluator = AccuracyEvaluator(
    X_train, y_train, X_val, y_val,
    model=model,
    cv_strategy=cv
)

# 5. Use in GA
ga = GeneticAlgorithm(evaluator=evaluator, ...)
```

## Best Practices

### Model Selection

**For quick experiments** (fast training):
- LogisticRegression
- KNN (small datasets)
- RandomForest (50 trees)

**For best performance** (slower):
- RandomForest (100-500 trees)
- XGBoost
- SVM with grid search

**For very large datasets**:
- Logistic Regression (scales well)
- SGDClassifier (online learning)

### Cross-Validation Trade-offs

| Strategy | Robustness | Speed | Use When |
|----------|------------|-------|----------|
| Single validation set | Low | Fast | Quick prototyping |
| 3-fold CV | Medium | Medium | Balanced |
| 5-fold CV | High | Slow | Production quality |
| 10-fold CV | Highest | Slowest | Small datasets |

## Extending

Add new model:

```python
# In models.py
SUPPORTED_MODELS = {
    'rf': RandomForestClassifier,
    'my_model': MyCustomClassifier  # Add here
}
```

Add new CV strategy:

```python
from fsga.ml.cv_strategy import CVStrategy

class MyCV(CVStrategy):
    def evaluate(self, model, X, y):
        # Your cross-validation logic
        return scores
```
