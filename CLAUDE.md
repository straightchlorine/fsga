# CLAUDE.md - AI Collaboration Context

This document provides context for AI assistants (Claude, GPT, etc.) working on this project.

---

## Project Genesis

**Created**: 2025-10-06
**Original Developer**: zweiss
**AI Assistant**: Claude (Anthropic) via Claude Code CLI

### Origin Story

This project was conceived during a conversation about creating a university ML project. The developer had an existing, well-structured genetic algorithm implementation for the knapsack problem ([~/code/knapsack-problem](../knapsack-problem/)) and wanted to:

1. Leverage that proven codebase
2. Create something ML-related for a university assignment
3. Build something "intricate" rather than basic

After analyzing the knapsack GA codebase (~2,424 lines of Python with clean OOP architecture), we decided on **Feature Selection via Genetic Algorithm** because:
- Reuses 80% of existing GA components (binary chromosomes, same operators)
- Clear ML connection (improves model performance)
- Intricate enough for academic distinction
- Fast to implement with existing foundation

---

## Architecture Philosophy

### Design Principles

1. **Code Reuse**: Adapt proven GA components from knapsack project rather than rebuild from scratch
2. **Modularity**: Each component (selector, operator, evaluator) is independent and swappable
3. **Extensibility**: Easy to add new operators, datasets, or ML models
4. **Production-Ready**: Not just a prototype - includes logging, config, testing, serialization
5. **Academic Rigor**: Statistical tests, comprehensive benchmarking, publication-quality visualizations

### Key Innovations Beyond Knapsack Project

| Component | Knapsack Version | FSGA Enhancement |
|-----------|------------------|------------------|
| **Fitness** | Maximize value within capacity constraint | Train ML model, return accuracy/F1 on validation set |
| **Chromosome** | Binary (1=item included) | Binary (1=feature selected) - same representation! |
| **Initialization** | Value-biased, weight-constrained | Correlation-biased, mutual information-based |
| **Selection** | Single-objective (maximize value) | Multi-objective (accuracy + sparsity via NSGA-II) |
| **Mutation** | Uniform bit-flip | Feature-aware (higher rate for correlated features) |
| **Operators** | Static rates | Adaptive (adjust based on population diversity) |
| **Evaluation** | O(n) calculation | O(n × train_time) - ML model training bottleneck |

---

## Project Structure Rationale

```
fsga/
├── core/           # GA engine - adapted from knapsack/genetic_algorithm.py, knapsack/population.py
├── operators/      # Crossover - ported from knapsack/operators/
├── mutations/      # Mutation - ported from knapsack/mutations/
├── selectors/      # Selection - ported from knapsack/selectors/ + new NSGA-II
├── evaluators/     # NEW: ML fitness functions (replaces knapsack/evaluators/fitness.py)
├── ml/             # NEW: Model wrappers, CV strategies, preprocessing
├── datasets/       # NEW: ML dataset loaders (replaces knapsack/dataset.py)
├── analysis/       # Enhanced from knapsack/analyze/
├── visualization/  # Enhanced from knapsack/visualization/
└── utils/          # Config, metrics, logging (new)
```

### Why This Structure?

- **Separation of Concerns**: GA logic (`core/`, `operators/`, etc.) is independent of ML logic (`ml/`, `evaluators/`)
- **Testability**: Each module can be tested in isolation
- **Reusability**: Operators work with any fitness function, evaluators work with any model
- **Academic Presentation**: Clear module boundaries make it easy to explain in reports/presentations

---

## Development Context

### Source Material

**Primary Reference**: `/home/zweiss/code/knapsack-problem/`
- Analyzed on 2025-10-06
- Key files to port:
  - `knapsack/genetic_algorithm.py` (225 lines) → `fsga/core/genetic_algorithm.py`
  - `knapsack/population.py` (219 lines) → `fsga/core/population.py`
  - `knapsack/operators/*.py` → `fsga/operators/*.py`
  - `knapsack/mutations/*.py` → `fsga/mutations/*.py`
  - `knapsack/selectors/*.py` → `fsga/selectors/*.py`

**Do NOT port**:
- `knapsack/dataset.py` - dataset logic is completely different (ML datasets vs. knapsack items)
- `knapsack/evaluators/fitness.py` - fitness calculation is ML-based, not value/capacity

### Current Status (as of 2025-10-06)

**Completed**:
- ✅ Directory structure created
- ✅ Main README.md written
- ✅ CLAUDE.md (this file) created
- ✅ PROJECT_PLAN.md with 20-day timeline

**In Progress**:
- ⏳ Module READMEs (writing now)
- ⏳ pyproject.toml setup

**Not Started**:
- ⬜ Core GA components (porting from knapsack)
- ⬜ ML integration (evaluators, model wrappers)
- ⬜ Experiments
- ⬜ Tests

---

## Code Style & Conventions

### Inherited from Knapsack Project

```python
# Class naming
class TournamentSelector(Selector):  # PascalCase for classes
    pass

# Method naming
def evaluate(self, chromosome):  # snake_case for methods
    pass

# Property pattern (used extensively in knapsack)
@property
def population_size(self):
    return self._population_size

@population_size.setter
def population_size(self, size):
    self._population_size = size
```

### New Conventions for FSGA

```python
# Type hints (add these - knapsack didn't have many)
def evaluate(self, chromosome: np.ndarray) -> float:
    pass

# Docstring format (use Google style)
def select_parents(self) -> tuple[np.ndarray, np.ndarray]:
    """Select two parents from the population.

    Returns:
        tuple: Two parent chromosomes as numpy arrays.

    Raises:
        ValueError: If population size is less than 2.
    """
```

### Ruff Configuration

Use the same ruff config from knapsack (`pyproject.toml`):
```toml
[tool.ruff]
line-length = 89
indent-width = 4

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

---

## Key Challenges & Solutions

### Challenge 1: Fitness Evaluation Performance

**Problem**: Training an ML model for every chromosome is expensive (O(n × train_time))

**Solutions**:
1. **Caching**: Hash chromosome → fitness mapping
2. **Parallel Evaluation**: Use `multiprocessing` to evaluate population in parallel
3. **Small Validation Set**: Use subset of data for fitness (not full training set)
4. **Fast Models**: Default to RandomForest (fast) instead of SVM (slow)

**Implementation Priority**: Start with naive approach, optimize later if needed

### Challenge 2: Multi-Objective Optimization (NSGA-II)

**Problem**: NSGA-II is complex (non-dominated sorting, crowding distance)

**Solutions**:
1. Use existing library (`pymoo`) if available
2. Implement simplified version focusing on 2 objectives only
3. Fallback: weighted sum approach (easier to implement)

**Implementation Priority**: Phase 3 (not MVP)

### Challenge 3: Feature Count = 0

**Problem**: Chromosome with all 0s is invalid (no features selected)

**Solutions**:
1. **Fitness Penalty**: Return 0 fitness if no features selected
2. **Repair Operator**: Force at least 1 feature to be 1
3. **Initialization**: Ensure at least 1 feature in initial population

**Chosen Approach**: #1 (simplest, lets GA learn)

---

## Testing Strategy

### Unit Tests

Each module needs tests:
```python
# tests/test_operators.py
def test_uniform_crossover_produces_valid_offspring():
    parent1 = np.array([1, 1, 0, 0])
    parent2 = np.array([0, 1, 1, 0])
    crossover = UniformCrossover()
    child1, child2 = crossover.crossover(parent1, parent2)
    assert all(gene in [0, 1] for gene in child1)
    assert all(gene in [0, 1] for gene in child2)
```

### Integration Tests

```python
# tests/test_end_to_end.py
def test_ga_runs_on_iris():
    X, y = load_iris(return_X_y=True)
    # ... run GA, assert fitness > threshold
```

### Coverage Target

80%+ coverage (knapsack had good coverage, maintain that standard)

---

## Common Pitfalls to Avoid

### From Knapsack Experience

1. **Don't mutate inputs**: Always copy chromosomes before modifying
   ```python
   # BAD
   def mutate(self, chromosome):
       chromosome[i] = 1 - chromosome[i]  # Modifies original!

   # GOOD
   def mutate(self, chromosome):
       mutated = chromosome.copy()
       mutated[i] = 1 - mutated[i]
       return mutated
   ```

2. **Validate population size**: Tournament size can't exceed population size
   ```python
   if self.tournament_size > len(self.population):
       raise ValueError("Tournament size too large")
   ```

3. **Handle edge cases**: Empty feature selection, single feature, etc.

### ML-Specific Pitfalls

1. **Data Leakage**: Never use test set for fitness evaluation
   ```python
   # BAD
   accuracy = model.score(X_test, y_test)  # Leakage!

   # GOOD
   accuracy = cross_val_score(model, X_train, y_train, cv=5).mean()
   ```

2. **Random Seeds**: Set seeds for reproducibility
   ```python
   model = RandomForestClassifier(random_state=42)
   np.random.seed(42)
   ```

3. **Class Imbalance**: Use balanced accuracy or F1 for imbalanced datasets

---

## Experiment Design

### Baseline Comparisons

Must compare against:
1. **All features** (no selection)
2. **Recursive Feature Elimination (RFE)** - sklearn wrapper method
3. **LASSO** (L1) - embedded method
4. **Mutual Information** - filter method

### Metrics to Report

| Metric | Purpose |
|--------|---------|
| Accuracy | Primary objective |
| F1-Score | For imbalanced datasets |
| # Features Selected | Sparsity measure |
| Computation Time | Efficiency |
| Stability | Feature selection consistency across runs |
| Statistical Significance | p-value from Wilcoxon test |

### Reproducibility Checklist

- [ ] Set random seeds (numpy, sklearn, Python)
- [ ] Save config files (YAML) with experiments
- [ ] Log all hyperparameters
- [ ] Version dataset splits (save train/test indices)
- [ ] Record library versions (`pip freeze`)

---

## Documentation Standards

### Module READMEs

Each module should have:
1. **Purpose**: What problem does this module solve?
2. **Components**: List of classes/functions
3. **Usage Example**: Minimal working code
4. **Extension Guide**: How to add new implementations

### Code Documentation

```python
class AccuracyEvaluator(Evaluator):
    """Evaluates feature subsets using classification accuracy.

    Trains a classifier on selected features and returns validation accuracy
    as the fitness score. Supports cross-validation for robust evaluation.

    Attributes:
        model: sklearn-compatible classifier
        X_train: Training feature matrix
        y_train: Training labels
        X_val: Validation feature matrix
        y_val: Validation labels
        cv_folds: Number of cross-validation folds (default: 5)

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier(random_state=42)
        >>> evaluator = AccuracyEvaluator(X_train, y_train, X_val, y_val, model)
        >>> fitness = evaluator.evaluate(chromosome)
    """
```

---

## Future Extensions (Post-MVP)

Ideas for enhancement after core functionality is complete:

1. **Deep Learning Integration**: Keras/PyTorch models
2. **Regression Support**: Currently classification-only
3. **Feature Engineering**: GA evolves feature transformations (log, sqrt, interactions)
4. **Ensemble GA**: Run multiple GAs, combine results
5. **Interactive Dashboard**: Streamlit app for real-time visualization
6. **Distributed Computing**: Ray/Dask for large-scale experiments
7. **AutoML Integration**: Compare against TPOT, Auto-sklearn

---

## AI Assistant Guidelines

When working on this project:

### Do's
✅ Maintain consistency with knapsack project structure
✅ Add type hints and comprehensive docstrings
✅ Write tests alongside implementation
✅ Use meaningful variable names (e.g., `chromosome`, not `x`)
✅ Preserve the property pattern from knapsack code
✅ Check if functionality exists in knapsack before reimplementing

### Don'ts
❌ Don't break existing abstractions (Evaluator, Selector, etc.)
❌ Don't add external dependencies without justification
❌ Don't sacrifice code clarity for minor performance gains
❌ Don't skip edge case handling (empty features, single sample, etc.)
❌ Don't modify knapsack source code (copy and adapt instead)

### When Porting from Knapsack

1. **Read the original implementation first** (understand before copying)
2. **Identify what changes** (dataset → ML data, fitness → model accuracy)
3. **Preserve the interface** (method signatures, property patterns)
4. **Update docstrings** (reflect ML context, not knapsack)
5. **Add ML-specific validation** (e.g., check X/y dimensions match)

### When Adding New Features

1. **Check PROJECT_PLAN.md** for design decisions
2. **Follow existing patterns** (see how similar features work)
3. **Update relevant READMEs** (module + main)
4. **Add tests** (unit + integration if applicable)
5. **Update CLAUDE.md** (document decisions made)

---

## Questions & Decisions Log

### Q1: Should we use pymoo for NSGA-II or implement from scratch?
**Decision**: Implement simplified 2-objective version from scratch first (educational value, no external dependency). Can switch to pymoo if complexity grows.

### Q2: How to handle categorical features?
**Decision**: Phase 1 only supports numerical features (one-hot encode upstream). Add categorical support in Phase 3 if needed.

### Q3: Cross-validation in fitness or separate validation set?
**Decision**: Validation set by default (faster). Cross-validation as optional parameter for evaluators.

### Q4: Store experiment results in database or files?
**Decision**: Files (JSON/pickle) for simplicity. Database (SQLite) if we need complex queries.

---

## Glossary (For AI Context)

| Term | Meaning in This Project |
|------|-------------------------|
| **Chromosome** | Binary array where `chromosome[i]=1` means "include feature i" |
| **Fitness** | ML model performance (accuracy/F1) on validation set |
| **Generation** | One iteration of the GA (selection → crossover → mutation) |
| **Population** | Collection of chromosomes (candidate feature subsets) |
| **Pareto Frontier** | Set of non-dominated solutions in multi-objective optimization |
| **Feature Sparsity** | Fraction of features NOT selected (1 - num_features/total_features) |
| **Wrapper Method** | Feature selection using ML model performance (what we're building) |
| **Filter Method** | Feature selection using statistical tests (mutual info, chi-squared) |
| **Embedded Method** | Feature selection built into model (LASSO, tree importance) |

---

## Contact & Collaboration

**Primary Developer**: zweiss (GitHub: @zweiss)
**AI Collaborator**: Claude (Anthropic) via Claude Code
**Project Start**: 2025-10-06
**Expected Completion**: 2025-10-26 (20-day timeline from PROJECT_PLAN.md)

For questions or collaboration:
- Open an issue on GitHub
- See PROJECT_PLAN.md for detailed roadmap
- Check module READMEs for component-specific docs

---

**Last Updated**: 2025-10-06
**Status**: Foundation phase (scaffolding complete, starting implementation)
