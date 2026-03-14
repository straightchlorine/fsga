# CLAUDE.md - AI Collaboration Context

This document provides context for AI assistants working on this project.

## Project Overview

University ML project: feature selection using a genetic algorithm. Adapted from a prior
[knapsack-problem](https://github.com/straightchlorine/knapsack-problem/) GA codebase,
reusing ~80% of the GA components (binary chromosomes, operators, selectors) and adding
ML fitness evaluation on top.

**Python**: >=3.13
**Dependencies**: numpy, pandas, scikit-learn, matplotlib, seaborn, pyyaml, scipy
**Tests**: pytest, 280+ tests, 82% coverage
**Linting**: ruff

---

## Project Structure

```
fsga/
├── core/           # GA engine (genetic_algorithm.py, population.py)
├── operators/      # Crossover: uniform, single-point, two-point, multi-point
├── mutations/      # Mutation: bitflip
├── selectors/      # Selection: tournament, roulette, ranking, elitism
├── evaluators/     # Fitness: accuracy, F1, balanced accuracy
├── ml/             # ModelWrapper (sklearn integration)
├── datasets/       # Dataset loaders (iris, wine, breast_cancer, digits)
├── analysis/       # Baselines + ExperimentRunner
├── visualization/  # 9 plot functions
└── utils/          # Config, metrics, logging, serialization
```

Other key paths:
- `tests/` -- 15 test files (pytest)
- `experiments/run_experiment.py` -- main experiment script
- `configs/` -- YAML configs (default, quick_test, high_dimensional, production)
- `report/` -- LaTeX report (untracked, not committed)
- `docs/` -- MkDocs documentation site

### Design

- **Strategy pattern**: operators, selectors, evaluators are all injected into GeneticAlgorithm
- **Separation of concerns**: GA logic is independent of ML logic
- Each module has its own README with usage examples and extension guide

### What changed from knapsack

| Component | Knapsack | FSGA |
|-----------|----------|------|
| Fitness | Maximize value within capacity | Train ML model, return accuracy/F1 |
| Chromosome | Binary (1=item included) | Binary (1=feature selected) -- same |
| Evaluation | O(n) calculation | O(n x train_time) -- ML training bottleneck |

---

## Code Style & Conventions

### Ruff configuration (pyproject.toml)

```toml
[tool.ruff]
line-length = 89
indent-width = 4
target-version = "py313"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### Naming

- PascalCase for classes: `TournamentSelector(Selector)`
- snake_case for methods/functions: `def evaluate(self, chromosome)`
- Property pattern from knapsack used throughout (`@property` + `@x.setter`)

### Type hints and docstrings

- Type hints on function signatures
- Google-style docstrings

---

## Testing

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=fsga --cov-report=html
```

- Unit tests for each operator/selector/evaluator
- Integration tests for full GA workflows
- Edge case tests (empty features, single sample, etc.)
- 80%+ coverage target

---

## Common Pitfalls

1. **Don't mutate inputs** -- always copy chromosomes before modifying
2. **Validate population size** -- tournament size can't exceed population size
3. **Handle edge cases** -- empty feature selection, single feature, etc.
4. **Data leakage** -- never use test set for fitness evaluation; use cross-validation on training set
5. **Random seeds** -- set seeds for reproducibility
6. **Class imbalance** -- use balanced accuracy or F1 for imbalanced datasets

---

## Experiment Design

### Baselines compared against

1. All features (no selection)
2. RFE (Recursive Feature Elimination)
3. LASSO (L1 regularization)
4. Mutual Information filter

### Metrics reported

- Accuracy, F1-Score
- Number of features selected
- Computation time
- Stability (Jaccard index across runs)
- Statistical significance (Wilcoxon test, Cohen's d)

### Tested results

| Dataset | GA Accuracy | Features Used | vs All Features |
|---------|-------------|---------------|-----------------|
| Breast Cancer | 98.3% | 12/30 (40%) | +2.6% |
| Iris | 98.3% | 2/4 (50%) | +6.2% |
| Wine | 100% | 6.5/13 (50%) | +1.4% |

---

## Guidelines for AI Assistants

### Do

- Maintain consistency with knapsack project structure
- Add type hints and docstrings
- Write tests alongside implementation
- Use meaningful variable names (`chromosome`, not `x`)
- Preserve the property pattern from knapsack code
- Check if functionality exists in knapsack before reimplementing

### Don't

- Don't break existing abstractions (Evaluator, Selector, etc.)
- Don't add external dependencies without justification
- Don't sacrifice code clarity for minor performance gains
- Don't skip edge case handling (empty features, single sample, etc.)

### When adding new features

1. Check docs/about/project-plan.md for design decisions
2. Follow existing patterns
3. Update relevant READMEs
4. Add tests (unit + integration if applicable)

---

## Glossary

| Term | Meaning |
|------|---------|
| **Chromosome** | Binary array where `chromosome[i]=1` means "include feature i" |
| **Fitness** | ML model performance (accuracy/F1) on validation set |
| **Generation** | One iteration of the GA (selection -> crossover -> mutation) |
| **Population** | Collection of chromosomes (candidate feature subsets) |
| **Feature Sparsity** | Fraction of features NOT selected |
| **Wrapper Method** | Feature selection using ML model performance (what this project does) |
| **Filter Method** | Feature selection using statistical tests (mutual info, chi-squared) |
| **Embedded Method** | Feature selection built into model (LASSO, tree importance) |
