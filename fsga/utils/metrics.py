"""Metrics for analyzing GA performance and stability.

Includes stability metrics (Jaccard index), effect size calculations (Cohen's d),
and convergence detection utilities.
"""

import numpy as np
from scipy import stats


def jaccard_similarity(set_a: np.ndarray, set_b: np.ndarray) -> float:
    """Calculate Jaccard similarity between two binary arrays.

    Jaccard similarity = |A ∩ B| / |A ∪ B|

    Args:
        set_a: Binary array (1=feature selected)
        set_b: Binary array (1=feature selected)

    Returns:
        float: Jaccard similarity (0.0 to 1.0)
            - 1.0 = identical feature sets
            - 0.0 = completely different sets

    Example:
        >>> a = np.array([1, 1, 0, 0, 1])
        >>> b = np.array([1, 0, 0, 1, 1])
        >>> jaccard_similarity(a, b)
        0.5  # 2 common features, 4 total unique features
    """
    if len(set_a) != len(set_b):
        raise ValueError("Arrays must have same length")

    # Convert to boolean
    set_a = set_a.astype(bool)
    set_b = set_b.astype(bool)

    intersection = np.sum(set_a & set_b)
    union = np.sum(set_a | set_b)

    # Handle edge case: both sets empty
    if union == 0:
        return 1.0

    return intersection / union


def jaccard_stability(chromosomes: list[np.ndarray]) -> float:
    """Calculate average pairwise Jaccard similarity for multiple chromosomes.

    Measures consistency of feature selection across multiple GA runs.

    Args:
        chromosomes: List of binary chromosomes from different runs

    Returns:
        float: Average Jaccard similarity (0.0 to 1.0)
            - 1.0 = all runs select same features
            - ~0.5 = moderate consistency
            - ~0.0 = random/inconsistent selection

    Example:
        >>> # Run GA 10 times
        >>> chromosomes = []
        >>> for _ in range(10):
        ...     results = ga.evolve()
        ...     chromosomes.append(results['best_chromosome'])
        >>> stability = jaccard_stability(chromosomes)
        >>> print(f"Stability: {stability:.2f}")
    """
    if len(chromosomes) < 2:
        raise ValueError("Need at least 2 chromosomes to calculate stability")

    similarities = []
    n = len(chromosomes)

    # Calculate pairwise similarities
    for i in range(n):
        for j in range(i + 1, n):
            sim = jaccard_similarity(chromosomes[i], chromosomes[j])
            similarities.append(sim)

    return float(np.mean(similarities))


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Calculate Cohen's d effect size between two groups.

    Cohen's d = (mean_a - mean_b) / pooled_std

    Effect size interpretation:
        - |d| < 0.2: Negligible
        - |d| < 0.5: Small
        - |d| < 0.8: Medium
        - |d| >= 0.8: Large

    Args:
        group_a: First group (e.g., GA accuracies)
        group_b: Second group (e.g., RFE accuracies)

    Returns:
        float: Cohen's d effect size
            Positive = group_a better than group_b

    Example:
        >>> ga_accuracies = np.array([0.95, 0.94, 0.96, 0.95])
        >>> rfe_accuracies = np.array([0.92, 0.91, 0.93, 0.92])
        >>> d = cohens_d(ga_accuracies, rfe_accuracies)
        >>> print(f"Effect size: {d:.2f} (large)" if abs(d) >= 0.8 else "")
    """
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)

    # Pooled standard deviation
    n_a = len(group_a)
    n_b = len(group_b)
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)

    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    # Avoid division by zero
    if pooled_std == 0:
        return 0.0

    return (mean_a - mean_b) / pooled_std


def effect_size_interpretation(d: float) -> str:
    """Get human-readable interpretation of Cohen's d.

    Args:
        d: Cohen's d value

    Returns:
        str: Interpretation ('negligible', 'small', 'medium', 'large')

    Example:
        >>> d = cohens_d(group_a, group_b)
        >>> print(effect_size_interpretation(d))
        'large'
    """
    abs_d = abs(d)

    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def convergence_detected(
    fitness_history: list[float] | np.ndarray,
    patience: int = 10,
    min_delta: float = 0.001,
) -> int | None:
    """Detect convergence in fitness history.

    Convergence = no improvement > min_delta for patience generations.

    Args:
        fitness_history: Best fitness per generation
        patience: Number of generations with no improvement
        min_delta: Minimum improvement threshold (relative)

    Returns:
        Optional[int]: Generation where convergence occurred, or None

    Example:
        >>> history = results['best_fitness_history']
        >>> conv_gen = convergence_detected(history, patience=10, min_delta=0.001)
        >>> if conv_gen:
        ...     print(f"Converged at generation {conv_gen}")
    """
    if len(fitness_history) < patience:
        return None

    fitness_history = np.array(fitness_history)

    for i in range(patience, len(fitness_history)):
        # Check if no improvement in last `patience` generations
        current_best = fitness_history[i]
        lookback_best = np.max(fitness_history[i - patience : i])

        improvement = (current_best - lookback_best) / (lookback_best + 1e-10)

        if improvement < min_delta:
            return i

    return None


def population_diversity(chromosomes: np.ndarray) -> float:
    """Calculate population diversity (average pairwise Hamming distance).

    Measures how different chromosomes are from each other.
    Higher diversity = more exploration, lower = convergence.

    Args:
        chromosomes: Population (shape: pop_size × num_features)

    Returns:
        float: Diversity score (0.0 to 1.0)
            - 1.0 = all chromosomes completely different
            - 0.0 = all chromosomes identical

    Example:
        >>> diversity = population_diversity(population.chromosomes)
        >>> print(f"Diversity: {diversity:.2%}")
    """
    if len(chromosomes) < 2:
        return 0.0

    n_pop, n_features = chromosomes.shape
    total_distance = 0
    count = 0

    # Average pairwise Hamming distance
    for i in range(n_pop):
        for j in range(i + 1, n_pop):
            distance = np.sum(chromosomes[i] != chromosomes[j]) / n_features
            total_distance += distance
            count += 1

    return total_distance / count if count > 0 else 0.0


def feature_selection_frequency(chromosomes: list[np.ndarray]) -> np.ndarray:
    """Calculate how often each feature is selected across runs.

    Args:
        chromosomes: List of binary chromosomes from multiple runs

    Returns:
        np.ndarray: Frequency of selection per feature (0.0 to 1.0)

    Example:
        >>> # Run GA 20 times
        >>> chromosomes = [ga.evolve()['best_chromosome'] for _ in range(20)]
        >>> frequencies = feature_selection_frequency(chromosomes)
        >>> print(f"Feature 0 selected in {frequencies[0]:.0%} of runs")
    """
    if len(chromosomes) == 0:
        raise ValueError("Need at least one chromosome")

    chromosomes = np.array(chromosomes)
    return np.mean(chromosomes, axis=0)


def core_features(
    chromosomes: list[np.ndarray],
    threshold: float = 0.8,
    feature_names: list[str] | None = None,
) -> list[int] | list[str]:
    """Get features selected in at least `threshold` fraction of runs.

    Args:
        chromosomes: List of binary chromosomes from multiple runs
        threshold: Minimum selection frequency (default: 0.8 = 80%)
        feature_names: Optional feature names for readable output

    Returns:
        list: Indices (or names) of core features

    Example:
        >>> chromosomes = [ga.evolve()['best_chromosome'] for _ in range(20)]
        >>> core = core_features(chromosomes, threshold=0.8, feature_names=names)
        >>> print(f"Core features: {core}")
        ['sepal_length', 'petal_width']
    """
    frequencies = feature_selection_frequency(chromosomes)
    core_indices = np.where(frequencies >= threshold)[0]

    if feature_names is not None:
        return [feature_names[i] for i in core_indices]
    else:
        return core_indices.tolist()


def sparsity(chromosome: np.ndarray) -> float:
    """Calculate feature sparsity (fraction of features NOT selected).

    Args:
        chromosome: Binary chromosome

    Returns:
        float: Sparsity (0.0 to 1.0)
            - 1.0 = no features selected
            - 0.0 = all features selected

    Example:
        >>> chromosome = np.array([1, 0, 0, 1, 0])  # 2/5 selected
        >>> sparsity(chromosome)
        0.6  # 3/5 not selected
    """
    n_features = len(chromosome)
    n_selected = np.sum(chromosome)
    return 1.0 - (n_selected / n_features)


def average_sparsity(chromosomes: list[np.ndarray]) -> float:
    """Calculate average sparsity across multiple chromosomes.

    Args:
        chromosomes: List of binary chromosomes

    Returns:
        float: Average sparsity

    Example:
        >>> chromosomes = [ga.evolve()['best_chromosome'] for _ in range(10)]
        >>> avg_sparsity = average_sparsity(chromosomes)
        >>> print(f"Average sparsity: {avg_sparsity:.1%}")
    """
    return float(np.mean([sparsity(c) for c in chromosomes]))


def wilcoxon_test(
    group_a: np.ndarray, group_b: np.ndarray, alternative: str = "two-sided"
) -> dict:
    """Perform Wilcoxon signed-rank test (paired, non-parametric).

    Tests if two paired samples come from same distribution.
    Use when data is paired (e.g., GA run i vs RFE run i on same split).

    Args:
        group_a: First group (e.g., GA accuracies)
        group_b: Second group (e.g., RFE accuracies)
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        dict: Test results with keys:
            - statistic: Test statistic
            - p_value: p-value
            - significant: True if p < 0.05

    Example:
        >>> ga_acc = np.array([0.95, 0.94, 0.96, 0.95, 0.94])
        >>> rfe_acc = np.array([0.92, 0.91, 0.93, 0.92, 0.91])
        >>> result = wilcoxon_test(ga_acc, rfe_acc, alternative='greater')
        >>> if result['significant']:
        ...     print("GA significantly better than RFE (p < 0.05)")
    """
    if len(group_a) != len(group_b):
        raise ValueError("Groups must have same size for paired test")

    statistic, p_value = stats.wilcoxon(
        group_a, group_b, alternative=alternative, zero_method="wilcox"
    )

    return {"statistic": statistic, "p_value": p_value, "significant": p_value < 0.05}


def mann_whitney_test(
    group_a: np.ndarray, group_b: np.ndarray, alternative: str = "two-sided"
) -> dict:
    """Perform Mann-Whitney U test (unpaired, non-parametric).

    Tests if two independent samples come from same distribution.
    Use when data is NOT paired (different train/test splits per run).

    Args:
        group_a: First group
        group_b: Second group
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        dict: Test results (statistic, p_value, significant)

    Example:
        >>> ga_acc = np.array([0.95, 0.94, 0.96])
        >>> baseline_acc = np.array([0.92, 0.91])
        >>> result = mann_whitney_test(ga_acc, baseline_acc)
    """
    statistic, p_value = stats.mannwhitneyu(
        group_a, group_b, alternative=alternative, method="auto"
    )

    return {"statistic": statistic, "p_value": p_value, "significant": p_value < 0.05}
