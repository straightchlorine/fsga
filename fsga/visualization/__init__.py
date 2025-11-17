"""Visualization module for GA results."""

from fsga.visualization.plots import (
    plot_accuracy_vs_sparsity,
    plot_combined_dashboard,
    plot_convergence,
    plot_diversity_evolution,
    plot_feature_count_comparison,
    plot_feature_frequency,
    plot_fitness_evolution,
    plot_method_comparison,
    plot_multi_metric_comparison,
)

__all__ = [
    "plot_fitness_evolution",
    "plot_diversity_evolution",
    "plot_convergence",
    "plot_feature_frequency",
    "plot_method_comparison",
    "plot_combined_dashboard",
    "plot_feature_count_comparison",
    "plot_multi_metric_comparison",
    "plot_accuracy_vs_sparsity",
]
