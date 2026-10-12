"""Smoke tests for fsga.visualization.

Release 1.1.8 shipped with `from itertools import zip_strict` in plots.py, which made
`import fsga.visualization` fail. Nothing imported the module in the test suite, so CI
stayed green. These tests import it and draw every plot once.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from fsga import visualization
from fsga.visualization import (
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


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def history():
    return np.linspace(0.6, 0.95, 12)


@pytest.fixture
def per_method_results():
    rng = np.random.default_rng(0)
    return {
        name: {
            "accuracies": rng.uniform(0.8, 0.95, 5),
            "n_features": n_features,
            "runtimes": rng.uniform(0.1, 1.0, 5),
            "stability": stability,
            "mean_n_features": float(n_features),
        }
        for name, n_features, stability in [
            ("GA", 3, 0.8),
            ("RFE", 5, 1.0),
            ("LASSO", 4, 1.0),
        ]
    }


def test_all_plot_functions_are_exported():
    for name in visualization.__all__:
        assert callable(getattr(visualization, name))


def test_fitness_evolution(history, tmp_path):
    out = tmp_path / "fitness.png"
    fig = plot_fitness_evolution(
        history, history - 0.1, history - 0.2, save_path=out, show=False
    )
    assert isinstance(fig, plt.Figure)
    assert out.stat().st_size > 0


def test_diversity_evolution(history):
    assert isinstance(plot_diversity_evolution(history, show=False), plt.Figure)


def test_convergence(history):
    assert isinstance(
        plot_convergence(history, convergence_gen=8, show=False), plt.Figure
    )


def test_feature_frequency():
    freq = np.array([0.1, 0.9, 0.7, 0.3])
    names = ["a", "b", "c", "d"]
    assert isinstance(plot_feature_frequency(freq, names, show=False), plt.Figure)


def test_method_comparison(tmp_path):
    out = tmp_path / "comparison.png"
    data = {"GA": [0.9, 0.92, 0.91], "RFE": [0.88, 0.9, 0.89]}
    plot_method_comparison(data, save_path=out, show=False)
    assert out.stat().st_size > 0


def test_combined_dashboard(history):
    freq = np.array([0.1, 0.9, 0.7, 0.3])
    assert isinstance(
        plot_combined_dashboard(history, history, freq, show=False), plt.Figure
    )


def test_feature_count_comparison(per_method_results):
    assert isinstance(
        plot_feature_count_comparison(per_method_results, show=False), plt.Figure
    )


def test_multi_metric_comparison(per_method_results, tmp_path):
    out = tmp_path / "multi.png"
    plot_multi_metric_comparison(per_method_results, save_path=out, show=False)
    assert out.stat().st_size > 0


def test_accuracy_vs_sparsity(per_method_results):
    fig = plot_accuracy_vs_sparsity(per_method_results, total_features=4, show=False)
    assert isinstance(fig, plt.Figure)
