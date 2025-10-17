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
