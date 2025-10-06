# fsga.datasets - Dataset Management

Loaders for ML benchmark datasets and utilities for train/test splitting.

## Components

### `loader.py`
Unified dataset loading interface.

```python
from fsga.datasets.loader import load_dataset

# Load from sklearn
X, y, feature_names = load_dataset('iris')

# Load with automatic train/test split
X_train, X_test, y_train, y_test, feature_names = load_dataset(
    'wine',
    split=True,
    test_size=0.2,
    stratify=True
)

# Load custom CSV
X, y, feature_names = load_dataset(
    'path/to/data.csv',
    target_column='label',
    feature_columns=['f1', 'f2', 'f3']  # Or None for all except target
)
```

### Available Datasets

| Name | Features | Classes | Samples | Source | Best For |
|------|----------|---------|---------|--------|----------|
| `iris` | 4 | 3 | 150 | sklearn | Quick testing |
| `wine` | 13 | 3 | 178 | sklearn | Small dataset experiments |
| `breast_cancer` | 30 | 2 | 569 | sklearn | Binary classification |
| `digits` | 64 | 10 | 1797 | sklearn | Multi-class |
| `ionosphere` | 34 | 2 | 351 | UCI | Feature selection benchmark |
| `sonar` | 60 | 2 | 208 | UCI | High noise |
| `madelon` | 500 | 2 | 2600 | UCI | High-dimensional (synthetic) |
| `mnist` | 784 | 10 | 70000 | sklearn | Scalability testing |

### `splitter.py`
Advanced train/validation/test splitting strategies.

```python
from fsga.datasets.splitter import DataSplitter

splitter = DataSplitter(
    test_size=0.2,
    val_size=0.1,  # Hold out 10% of training for validation
    stratify=True,
    random_state=42
)

X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)
```

**Strategies**:
- `train_test`: Simple 80/20 split
- `train_val_test`: 70/10/20 split (recommended for GA)
- `k_fold`: Returns k train/val pairs for cross-validation
- `time_series`: No shuffling (for temporal data)

### `synthetic.py`
Generate synthetic datasets for controlled experiments.

```python
from fsga.datasets.synthetic import generate_classification_data

X, y = generate_classification_data(
    n_samples=1000,
    n_features=50,
    n_informative=10,  # Only 10 features are predictive
    n_redundant=5,     # 5 are linear combinations of informative
    n_repeated=5,      # 5 are duplicates
    n_classes=2,
    class_sep=1.0,     # How separable are classes
    random_state=42
)

# Perfect for testing: we know which features should be selected!
```

**Use cases**:
- Validate GA can find the 10 informative features
- Test performance with varying noise levels
- Scalability testing (generate 10,000 features)

## Usage Patterns

### Pattern 1: Quick Experiment on Benchmark
```python
from fsga.datasets.loader import load_dataset

# One-liner with split
X_train, X_test, y_train, y_test, names = load_dataset('wine', split=True)
```

### Pattern 2: Controlled Synthetic Experiment
```python
from fsga.datasets.synthetic import generate_classification_data

# Generate data where we KNOW 10/100 features are informative
X, y = generate_classification_data(
    n_features=100,
    n_informative=10,
    random_state=42
)

# Run GA, check if it selects ~10 features
```

### Pattern 3: Custom Dataset
```python
import pandas as pd
from fsga.datasets.loader import DatasetLoader

df = pd.read_csv('my_data.csv')
loader = DatasetLoader()
X, y, feature_names = loader.from_dataframe(
    df,
    target='outcome',
    drop_columns=['id', 'timestamp']  # Non-feature columns
)
```

## Dataset Properties

### Accessing Metadata
```python
from fsga.datasets.loader import get_dataset_info

info = get_dataset_info('madelon')
print(info['n_features'])      # 500
print(info['n_classes'])       # 2
print(info['n_samples'])       # 2600
print(info['task'])            # 'binary_classification'
print(info['imbalanced'])      # False
```

### Feature Names
```python
X, y, feature_names = load_dataset('wine')
print(feature_names)
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', ...]

# Use in results
selected_indices = np.where(best_chromosome == 1)[0]
selected_features = feature_names[selected_indices]
print(f"Selected: {selected_features}")
```

## Extending

### Add New Dataset Loader
```python
# In loader.py
def _load_my_dataset(self):
    """Load custom dataset from source."""
    # Your loading logic
    X = ...
    y = ...
    feature_names = [...]
    return X, y, feature_names

# Register
LOADERS = {
    'iris': _load_iris,
    'wine': _load_wine,
    'my_dataset': _load_my_dataset  # Add here
}
```

### Download from OpenML
```python
from sklearn.datasets import fetch_openml

def load_openml_dataset(dataset_id):
    data = fetch_openml(data_id=dataset_id, as_frame=True)
    return data.data.values, data.target.values, data.feature_names
```

## Best Practices

1. **Always stratify** for imbalanced datasets
2. **Set random_state** for reproducibility
3. **Use validation set** separate from test set (GA uses val, report test)
4. **Scale features** before use (see fsga.ml.preprocessor)
5. **Check for missing values** (handle or remove)
