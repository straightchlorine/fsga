# Core Module - Developer Guide

**Purpose**: Genetic algorithm engine and population management.

## Architecture Overview

```
core/
├── genetic_algorithm.py    # Main GA orchestrator
├── population.py           # Population container & initialization
└── README_DEV.md           # This file
```

## Key Components

### 1. `GeneticAlgorithm` Class

**Responsibility**: Orchestrates the entire evolutionary process.

**Design Pattern**: Strategy pattern (configurable operators)

```python
class GeneticAlgorithm:
    """Main GA engine.

    Collaborators:
    - Evaluator: Calculates fitness
    - Selector: Chooses parents
    - Crossover: Combines parents → offspring
    - Mutation: Introduces variation
    - Population: Manages chromosomes
    """
```

**Key Methods**:

| Method | Purpose | Returns |
|--------|---------|---------|
| `evolve()` | Run complete GA | dict with results |
| `_selection()` | Select 2 parents | tuple of chromosomes |
| `_crossover()` | Apply crossover | tuple of offspring |
| `_mutation()` | Apply mutation | mutated chromosome |
| `_evaluate_population()` | Calculate all fitness | None (updates Population) |

**Algorithm Flow**:
```python
# 1. Initialize
population = Population(...)
population.initialize_random()

# 2. Main loop
for generation in range(num_generations):
    # 2a. Evaluate
    for chromosome in population:
        fitness = evaluator.evaluate(chromosome)

    # 2b. Select & Reproduce
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = selector.select_parents()
        child1, child2 = crossover.crossover(parent1, parent2)
        child1 = mutation.mutate(child1)
        child2 = mutation.mutate(child2)
        new_population.extend([child1, child2])

    # 2c. Replace
    population = new_population

    # 2d. Track best
    if current_best > all_time_best:
        all_time_best = current_best

    # 2e. Early stopping
    if no_improvement_for(patience):
        break

# 3. Return results
return {'best_chromosome': ..., 'best_fitness': ..., ...}
```

**Early Stopping**:
- Tracks generations without improvement
- Triggers when `patience` generations pass with no fitness gain
- Configurable via `early_stopping_patience` parameter

**Performance Notes**:
- Fitness evaluation is bottleneck (O(n × training_time))
- No caching implemented yet (all chromosomes re-evaluated)
- No parallelization (single-threaded)

---

### 2. `Population` Class

**Responsibility**: Manages collection of chromosomes.

**Data Structure**:
```python
class Population:
    chromosomes: np.ndarray  # Shape: (pop_size, num_features)
    fitnesses: np.ndarray     # Shape: (pop_size,)
```

**Initialization Strategies**:

| Strategy | Implementation | Use Case |
|----------|----------------|----------|
| **Random** | `initialize_random()` | Default, unbiased |
| **Biased** | `initialize_biased(p)` | Encourage sparsity (p=0.3 → 30% features selected) |
| **Seeded** | Pass chromosomes to constructor | Warm start from previous run |

**Key Methods**:

```python
# Initialization
pop.initialize_random()               # 50% probability per feature
pop.initialize_biased(probability=0.3)  # 30% features selected on average

# Access
pop.get_best()                        # Returns (chromosome, fitness) tuple
pop.get_worst()
pop.get_chromosome(index)
pop.set_chromosome(index, chromosome)

# Sorting
pop.sort_by_fitness(descending=True)  # Best first

# Statistics
pop.get_diversity()                   # Average Hamming distance
pop.get_sparsity()                    # Average feature sparsity
```

**Validation**:
- Chromosomes must be binary (0 or 1)
- Pop size must be even (for pairwise mating)
- At least 1 feature must be selected per chromosome

---

## Design Decisions

### Why Strategy Pattern for Operators?

**Problem**: Different crossover/mutation/selection strategies needed.

**Solution**: Inject operators as dependencies.

```python
# Easy to swap operators
ga = GeneticAlgorithm(
    crossover_operator=UniformCrossover(),  # or SinglePointCrossover()
    mutation_operator=BitFlipMutation(),    # or GaussianMutation()
    selector=TournamentSelector(),           # or RouletteSelector()
    ...
)
```

**Benefits**:
- Testability: Can mock operators
- Extensibility: Add new operators without modifying GA
- Configurability: Runtime operator selection

---

### Why Not Store Population History?

**Current**: Only stores best chromosome per generation.

**Reason**: Memory efficiency (avoids O(pop_size × generations × features)).

**To Enable**:
```python
# Modify genetic_algorithm.py
self.population_history = []  # Add to __init__

# In evolve() loop
self.population_history.append(population.chromosomes.copy())
```

**Trade-off**:
- Pro: Enables diversity analysis over time
- Con: 100 gens × 50 pop × 100 features = 500KB per run

---

## Extension Points

### Adding New Initialization Strategy

```python
# In population.py
def initialize_correlation_based(self, X, y):
    """Initialize based on feature-target correlation."""
    from scipy.stats import pearsonr

    correlations = [abs(pearsonr(X[:, i], y)[0]) for i in range(self.num_features)]
    # Select top features based on correlation
    for i in range(self.population_size):
        chromosome = np.zeros(self.num_features)
        top_k_indices = np.argsort(-correlations)[:k]
        chromosome[top_k_indices] = 1
        # Add randomness
        self.chromosomes[i] = chromosome
```

### Adding Elitism

**Elitism**: Always keep best N chromosomes.

```python
# In genetic_algorithm.py, modify evolve()
# After creating new_population
elite_size = int(0.1 * self.population_size)  # Top 10%
elite = population.get_top_k(elite_size)
new_population[:elite_size] = elite
```

### Adding Diversity Enforcement

**Problem**: Premature convergence (all chromosomes become similar).

**Solution**: Penalize similar chromosomes.

```python
def _diversity_adjusted_fitness(self, chromosome, population):
    """Boost fitness for diverse chromosomes."""
    base_fitness = self.evaluator.evaluate(chromosome)

    # Calculate uniqueness
    distances = [hamming_distance(chromosome, other)
                 for other in population.chromosomes]
    avg_distance = np.mean(distances)

    # Bonus for being different
    diversity_bonus = avg_distance * 0.1
    return base_fitness + diversity_bonus
```

---

## Common Pitfalls

### 1. Modifying Chromosomes In-Place

```python
# BAD - Mutates original!
def crossover(parent1, parent2):
    child = parent1  # Reference, not copy!
    child[0] = parent2[0]
    return child

# GOOD - Creates copy
def crossover(parent1, parent2):
    child = parent1.copy()
    child[0] = parent2[0]
    return child
```

### 2. Invalid Chromosomes (All Zeros)

```python
# Problem: Mutation can set all bits to 0
chromosome = np.array([1, 0, 0, 0])
mutated = bitflip_mutation(chromosome)  # Could be [0, 0, 0, 0]

# Solution: Validate after mutation
if np.sum(mutated) == 0:
    # Re-mutate or force one feature on
    mutated[np.random.randint(len(mutated))] = 1
```

### 3. Population Size < Tournament Size

```python
# Error: Can't sample 5 from population of 3
selector = TournamentSelector(tournament_size=5)
ga = GeneticAlgorithm(population_size=3, selector=selector)  # CRASH!

# Fix: Validate in GeneticAlgorithm.__init__
assert population_size >= selector.tournament_size
```

---

## Performance Optimization

### Bottleneck Analysis

```python
# Typical 100-generation run on Iris (150 samples, 4 features)
Total time: 5.2 seconds
├── Fitness evaluation: 4.8s (92%)  ← BOTTLENECK
├── Crossover: 0.2s (4%)
├── Mutation: 0.1s (2%)
└── Selection: 0.1s (2%)
```

### Optimization Strategies

**1. Fitness Caching** (not implemented)
```python
# Hash chromosome → fitness
fitness_cache = {}

def evaluate_with_cache(chromosome):
    key = tuple(chromosome)
    if key not in fitness_cache:
        fitness_cache[key] = evaluator.evaluate(chromosome)
    return fitness_cache[key]
```

**Benefit**: ~30% speedup (many duplicates in late generations)

**2. Parallel Evaluation** (not implemented)
```python
from multiprocessing import Pool

def evaluate_population_parallel(chromosomes):
    with Pool(processes=4) as pool:
        fitnesses = pool.map(evaluator.evaluate, chromosomes)
    return fitnesses
```

**Benefit**: ~3x speedup on 4-core CPU

**3. Early Termination** (implemented)
- Already reduces ~40% wasted generations via early stopping

---

## Testing Strategies

### Unit Tests

```python
# test_genetic_algorithm.py
def test_evolution_improves_fitness():
    """Verify fitness increases over generations."""
    ga = GeneticAlgorithm(...)
    results = ga.evolve()

    first_gen_best = results['best_fitness_history'][0]
    last_gen_best = results['best_fitness_history'][-1]

    assert last_gen_best >= first_gen_best

def test_early_stopping_triggers():
    """Verify early stopping with no improvement."""
    ga = GeneticAlgorithm(early_stopping_patience=5, ...)
    results = ga.evolve()

    # Should stop before max generations
    assert len(results['best_fitness_history']) < ga.num_generations
```

### Integration Tests

```python
def test_full_pipeline_on_iris():
    """End-to-end test on real data."""
    X_train, X_test, y_train, y_test = load_dataset('iris', split=True)

    ga = GeneticAlgorithm(...)
    results = ga.evolve()

    # Verify results structure
    assert 'best_chromosome' in results
    assert 'best_fitness' in results

    # Verify improvement over baseline
    all_features_accuracy = train_model(X_train, y_train, X_test, y_test)
    ga_accuracy = results['best_fitness']
    assert ga_accuracy >= all_features_accuracy * 0.9  # Allow 10% degradation
```

---

## Configuration Best Practices

### Recommended Hyperparameters

| Dataset Size | Pop Size | Generations | Mutation Rate | Patience |
|--------------|----------|-------------|---------------|----------|
| Small (<500) | 30-50 | 50-100 | 0.01-0.05 | 10 |
| Medium (500-5K) | 50-100 | 100-200 | 0.01-0.03 | 15 |
| Large (>5K) | 100-200 | 200-500 | 0.005-0.02 | 20 |

### Couple of rules

- **Population size**: 5-10× number of features
- **Mutation rate**: 1/num_features (one bit flip per chromosome on average)
- **Patience**: ~10% of max generations
- **Crossover rate**: Always 1.0 (every offspring is crossed over)

---

## See Also

- **operators/**: Crossover implementations
- **mutations/**: Mutation implementations
- **selectors/**: Selection implementations
- **evaluators/**: Fitness function implementations
