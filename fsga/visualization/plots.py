"""Visualization functions for GA results and analysis.

Provides plotting utilities for fitness evolution, population diversity,
feature stability, and comparison across methods.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_fitness_evolution(
    best_fitness_history: list[float] | np.ndarray,
    avg_fitness_history: Optional[list[float] | np.ndarray] = None,
    worst_fitness_history: Optional[list[float] | np.ndarray] = None,
    title: str = "Fitness Evolution",
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot fitness evolution over generations.

    Args:
        best_fitness_history: Best fitness per generation
        avg_fitness_history: Average fitness per generation (optional)
        worst_fitness_history: Worst fitness per generation (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib.figure.Figure: The created figure

    Example:
        >>> results = ga.evolve()
        >>> fig = plot_fitness_evolution(
        ...     results['best_fitness_history'],
        ...     results.get('avg_fitness_history'),
        ...     title="GA on Iris Dataset"
        ... )
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    generations = range(len(best_fitness_history))

    # Plot best fitness
    ax.plot(generations, best_fitness_history, label="Best", linewidth=2, color="green")

    # Plot average fitness if provided
    if avg_fitness_history is not None:
        ax.plot(
            generations, avg_fitness_history, label="Average", linewidth=2, color="blue"
        )

    # Plot worst fitness if provided
    if worst_fitness_history is not None:
        ax.plot(
            generations, worst_fitness_history, label="Worst", linewidth=2, color="red"
        )

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_diversity_evolution(
    diversity_history: list[float] | np.ndarray,
    title: str = "Population Diversity Evolution",
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot population diversity over generations.

    Args:
        diversity_history: Diversity metric per generation
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib.figure.Figure: The created figure

    Example:
        >>> from fsga.utils.metrics import population_diversity
        >>> diversity_history = []
        >>> for gen, pop in enumerate(ga_history):
        ...     diversity_history.append(population_diversity(pop))
        >>> fig = plot_diversity_evolution(diversity_history)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    generations = range(len(diversity_history))
    ax.plot(generations, diversity_history, linewidth=2, color="purple")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Diversity", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add horizontal line at convergence threshold (low diversity)
    ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Low diversity")
    ax.legend(loc="best")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_convergence(
    fitness_history: list[float] | np.ndarray,
    convergence_gen: Optional[int] = None,
    patience: int = 10,
    title: str = "Convergence Detection",
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot fitness with convergence point marked.

    Args:
        fitness_history: Best fitness per generation
        convergence_gen: Generation where convergence detected (optional)
        patience: Patience parameter used for early stopping
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib.figure.Figure: The created figure

    Example:
        >>> from fsga.utils.metrics import convergence_detected
        >>> history = results['best_fitness_history']
        >>> conv_gen = convergence_detected(history, patience=10)
        >>> fig = plot_convergence(history, conv_gen, patience=10)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    generations = range(len(fitness_history))
    ax.plot(generations, fitness_history, linewidth=2, color="green")

    if convergence_gen is not None:
        # Mark convergence point
        ax.axvline(
            x=convergence_gen,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Converged at gen {convergence_gen}",
        )
        ax.scatter(
            [convergence_gen],
            [fitness_history[convergence_gen]],
            color="red",
            s=100,
            zorder=5,
        )

        # Shade the patience window
        if convergence_gen >= patience:
            ax.axvspan(
                convergence_gen - patience,
                convergence_gen,
                alpha=0.2,
                color="yellow",
                label=f"Patience window ({patience} gens)",
            )

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_feature_frequency(
    frequencies: np.ndarray,
    feature_names: Optional[list[str]] = None,
    threshold: float = 0.5,
    title: str = "Feature Selection Frequency",
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot feature selection frequency across multiple runs.

    Args:
        frequencies: Selection frequency per feature (0.0 to 1.0)
        feature_names: Feature names for x-axis labels (optional)
        threshold: Threshold line for "core features"
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib.figure.Figure: The created figure

    Example:
        >>> from fsga.utils.metrics import feature_selection_frequency
        >>> chromosomes = [run['best_chromosome'] for run in multiple_runs]
        >>> frequencies = feature_selection_frequency(chromosomes)
        >>> fig = plot_feature_frequency(frequencies, feature_names=names)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    n_features = len(frequencies)
    x = np.arange(n_features)

    if feature_names is None:
        feature_names = [f"F{i}" for i in range(n_features)]

    # Color bars above/below threshold differently
    colors = ["green" if f >= threshold else "gray" for f in frequencies]
    ax.bar(x, frequencies, color=colors, alpha=0.7)

    # Add threshold line
    ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Core threshold ({threshold:.0%})",
    )

    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel("Selection Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_method_comparison(
    results_dict: dict[str, list[float]],
    metric_name: str = "Accuracy",
    title: str = "Method Comparison",
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot box plot comparing multiple methods.

    Args:
        results_dict: Dictionary mapping method names to lists of scores
        metric_name: Name of the metric being compared
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib.figure.Figure: The created figure

    Example:
        >>> results = {
        ...     'GA': [0.95, 0.94, 0.96, 0.95],
        ...     'RFE': [0.92, 0.91, 0.93, 0.92],
        ...     'LASSO': [0.90, 0.89, 0.91, 0.90],
        ...     'All Features': [0.88, 0.87, 0.89, 0.88]
        ... }
        >>> fig = plot_method_comparison(results, metric_name="Accuracy")
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(results_dict.keys())
    data = [results_dict[method] for method in methods]

    bp = ax.boxplot(data, labels=methods, patch_artist=True, showmeans=True)

    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_combined_dashboard(
    best_fitness_history: list[float] | np.ndarray,
    diversity_history: list[float] | np.ndarray,
    frequencies: np.ndarray,
    feature_names: Optional[list[str]] = None,
    title: str = "GA Dashboard",
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Create a combined dashboard with multiple plots.

    Args:
        best_fitness_history: Best fitness per generation
        diversity_history: Diversity per generation
        frequencies: Feature selection frequencies
        feature_names: Feature names (optional)
        title: Overall title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib.figure.Figure: The created figure

    Example:
        >>> fig = plot_combined_dashboard(
        ...     results['best_fitness_history'],
        ...     diversity_history,
        ...     frequencies,
        ...     feature_names=names
        ... )
    """
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Fitness evolution (top)
    ax1 = plt.subplot(2, 2, 1)
    generations = range(len(best_fitness_history))
    ax1.plot(generations, best_fitness_history, linewidth=2, color="green")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Fitness")
    ax1.set_title("Fitness Evolution")
    ax1.grid(True, alpha=0.3)

    # Diversity evolution (top right)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(range(len(diversity_history)), diversity_history, linewidth=2, color="purple")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Diversity")
    ax2.set_title("Population Diversity")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.1, color="red", linestyle="--", alpha=0.5)

    # Feature frequencies (bottom, spanning both columns)
    ax3 = plt.subplot(2, 1, 2)
    n_features = len(frequencies)
    x = np.arange(n_features)
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(n_features)]
    colors = ["green" if f >= 0.5 else "gray" for f in frequencies]
    ax3.bar(x, frequencies, color=colors, alpha=0.7)
    ax3.axhline(y=0.5, color="red", linestyle="--", linewidth=2)
    ax3.set_xlabel("Features")
    ax3.set_ylabel("Selection Frequency")
    ax3.set_title("Feature Selection Frequency")
    ax3.set_xticks(x)
    ax3.set_xticklabels(feature_names, rotation=45, ha="right")
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_feature_count_comparison(
    results_dict: dict[str, dict],
    metric_name: str = "Number of Features",
    title: str = "Feature Count Comparison",
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot feature count comparison across methods.

    Args:
        results_dict: Dictionary mapping method names to result dicts
                     (must contain 'n_features' or 'mean_n_features')
        metric_name: Name of the metric being compared
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib.figure.Figure: The created figure

    Example:
        >>> results = {
        ...     'GA': {'n_features': np.array([12, 13, 11, 12])},
        ...     'RFE': {'n_features': np.array([15, 15, 15, 15])},
        ...     'All Features': {'n_features': 30}
        ... }
        >>> fig = plot_feature_count_comparison(results)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(results_dict.keys())
    data = []

    for method in methods:
        result = results_dict[method]
        # Handle both array and scalar n_features
        if "n_features" in result:
            n_feat = result["n_features"]
            if isinstance(n_feat, (int, float)):
                # Scalar (e.g., "All Features")
                data.append([n_feat])
            else:
                # Array (multiple runs)
                data.append(n_feat)
        elif "mean_n_features" in result:
            # Fallback to mean if only mean available
            data.append([result["mean_n_features"]])
        else:
            raise ValueError(f"Method '{method}' missing n_features data")

    bp = ax.boxplot(data, labels=methods, patch_artist=True, showmeans=True)

    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_multi_metric_comparison(
    results_dict: dict[str, dict],
    title: str = "Multi-Metric Comparison",
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot 2x2 grid comparing multiple metrics across methods.

    Args:
        results_dict: Dictionary mapping method names to result dicts
                     (must contain accuracies, n_features, runtimes, stability)
        title: Overall title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib.figure.Figure: The created figure

    Example:
        >>> results = {
        ...     'GA': {
        ...         'accuracies': np.array([0.95, 0.94, 0.96]),
        ...         'n_features': np.array([12, 13, 11]),
        ...         'runtimes': np.array([2.1, 2.3, 2.0]),
        ...         'stability': 0.85
        ...     },
        ...     'RFE': {...}
        ... }
        >>> fig = plot_multi_metric_comparison(results)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    methods = list(results_dict.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

    # 1. Accuracy (top-left)
    ax1 = axes[0, 0]
    acc_data = [results_dict[m]["accuracies"] for m in methods]
    bp1 = ax1.boxplot(acc_data, labels=methods, patch_artist=True, showmeans=True)
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Classification Accuracy", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.tick_params(axis="x", rotation=15)

    # 2. Feature Count (top-right)
    ax2 = axes[0, 1]
    feat_data = []
    for m in methods:
        n_feat = results_dict[m].get("n_features")
        if isinstance(n_feat, (int, float)):
            feat_data.append([n_feat])
        else:
            feat_data.append(n_feat)
    bp2 = ax2.boxplot(feat_data, labels=methods, patch_artist=True, showmeans=True)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel("Number of Features", fontsize=11)
    ax2.set_title("Feature Count (Sparsity)", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.tick_params(axis="x", rotation=15)

    # 3. Runtime (bottom-left)
    ax3 = axes[1, 0]
    runtime_data = [results_dict[m]["runtimes"] for m in methods]
    bp3 = ax3.boxplot(runtime_data, labels=methods, patch_artist=True, showmeans=True)
    for patch, color in zip(bp3["boxes"], colors):
        patch.set_facecolor(color)
    ax3.set_ylabel("Runtime (seconds)", fontsize=11)
    ax3.set_title("Computational Efficiency", fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.tick_params(axis="x", rotation=15)

    # 4. Stability (bottom-right) - bar chart
    ax4 = axes[1, 1]
    stability_data = [results_dict[m].get("stability", 0) for m in methods]
    bars = ax4.bar(methods, stability_data, color=colors, alpha=0.7)
    ax4.set_ylabel("Jaccard Stability", fontsize=11)
    ax4.set_title("Feature Selection Stability", fontweight="bold")
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.tick_params(axis="x", rotation=15)

    # Add value labels on bars
    for bar, val in zip(bars, stability_data):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_accuracy_vs_sparsity(
    results_dict: dict[str, dict],
    total_features: int,
    title: str = "Accuracy vs. Sparsity Trade-off",
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot accuracy vs sparsity scatter plot.

    Shows the trade-off between accuracy and feature reduction.
    Goal: Top-right corner (high accuracy, high sparsity).

    Args:
        results_dict: Dictionary mapping method names to result dicts
        total_features: Total number of features in dataset
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib.figure.Figure: The created figure

    Example:
        >>> fig = plot_accuracy_vs_sparsity(results, total_features=30)
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    methods = list(results_dict.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

    for i, (method, results) in enumerate(results_dict.items()):
        accuracies = results["accuracies"]
        n_features = results.get("n_features")

        if isinstance(n_features, (int, float)):
            # Scalar
            sparsity = 1 - (n_features / total_features)
            ax.scatter(
                [sparsity],
                [accuracies.mean()],
                s=200,
                color=colors[i],
                alpha=0.7,
                label=method,
                edgecolors="black",
                linewidth=1.5,
            )
            # Error bars for accuracy
            if len(accuracies) > 1:
                ax.errorbar(
                    [sparsity],
                    [accuracies.mean()],
                    yerr=[accuracies.std()],
                    fmt="none",
                    color=colors[i],
                    alpha=0.5,
                    capsize=5,
                )
        else:
            # Array - plot all points
            sparsities = 1 - (n_features / total_features)
            ax.scatter(
                sparsities,
                accuracies,
                s=150,
                color=colors[i],
                alpha=0.6,
                label=method,
                edgecolors="black",
                linewidth=1,
            )
            # Plot mean
            ax.scatter(
                [sparsities.mean()],
                [accuracies.mean()],
                s=300,
                color=colors[i],
                alpha=1.0,
                edgecolors="black",
                linewidth=2,
                marker="D",
            )

    ax.set_xlabel("Sparsity (1 - #features/#total)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add ideal region annotation
    ax.axhline(y=0.95, color="green", linestyle="--", alpha=0.3, linewidth=1)
    ax.axvline(x=0.5, color="green", linestyle="--", alpha=0.3, linewidth=1)
    ax.text(
        0.98,
        0.02,
        "Goal: Top-right\n(High accuracy, High sparsity)",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(
        max(0, min([r["accuracies"].min() for r in results_dict.values()]) - 0.05), 1.02
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig
