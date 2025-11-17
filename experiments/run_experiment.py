#!/usr/bin/env python3
"""Main experiment runner for GA feature selection.

Unified script that runs comprehensive experiments with full visualization suite.
Supports both quick testing and full analysis modes.

Usage:
    python run_experiment.py                      # Full analysis (default)
    python run_experiment.py --quick              # Quick test mode
    python run_experiment.py --datasets iris wine # Specific datasets only
    python run_experiment.py --no-plots           # Skip visualization generation
"""

import argparse
import logging
from pathlib import Path

from fsga.analysis.experiment_runner import ExperimentRunner
from fsga.datasets.loader import load_dataset
from fsga.utils.metrics import feature_selection_frequency
from fsga.visualization import (
    plot_accuracy_vs_sparsity,
    plot_combined_dashboard,
    plot_feature_count_comparison,
    plot_fitness_evolution,
    plot_method_comparison,
    plot_multi_metric_comparison,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiment.log')
    ]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "datasets": ["iris", "wine", "breast_cancer"],
    "model_type": "rf",
    "n_runs": 10,
    "random_state": 42,
    "population_size": 50,
    "num_generations": 100,
    "mutation_rate": 0.01,
    "early_stopping_patience": 10,
}

QUICK_CONFIG = {
    **DEFAULT_CONFIG,
    "datasets": ["iris"],
    "n_runs": 3,
    "num_generations": 30,
}


def generate_visualizations(runner, dataset_name, total_features, plots_dir):
    """Generate complete visualization suite for a dataset.

    Args:
        runner: ExperimentRunner with results
        dataset_name: Dataset name
        total_features: Total number of features in dataset
        plots_dir: Directory to save plots
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating visualizations for {dataset_name}")

    # 1. Method Comparison (Accuracy)
    logger.info("  [1/6] Method comparison (accuracy)...")
    method_accuracies = {
        method: results["accuracies"] for method, results in runner.results.items()
    }
    plot_method_comparison(
        method_accuracies,
        metric_name="Accuracy",
        title=f"Method Comparison: {dataset_name.title()}",
        save_path=plots_dir / f"{dataset_name}_accuracy_comparison.png",
        show=False,
    )

    # 2. Feature Count Comparison
    logger.info("  [2/6] Feature count comparison...")
    plot_feature_count_comparison(
        runner.results,
        metric_name="Number of Features Selected",
        title=f"Feature Count Comparison: {dataset_name.title()}",
        save_path=plots_dir / f"{dataset_name}_feature_count_comparison.png",
        show=False,
    )

    # 3. Multi-Metric Comparison
    logger.info("  [3/6] Multi-metric comparison...")
    plot_multi_metric_comparison(
        runner.results,
        title=f"Comprehensive Comparison: {dataset_name.title()}",
        save_path=plots_dir / f"{dataset_name}_multi_metric.png",
        show=False,
    )

    # 4. Accuracy vs Sparsity
    logger.info("  [4/6] Accuracy vs sparsity trade-off...")
    plot_accuracy_vs_sparsity(
        runner.results,
        total_features=total_features,
        title=f"Accuracy vs. Sparsity: {dataset_name.title()}",
        save_path=plots_dir / f"{dataset_name}_pareto.png",
        show=False,
    )

    # 5. GA Fitness Evolution
    if "GA" in runner.results and "fitness_histories" in runner.results["GA"]:
        logger.info("  [5/6] GA fitness evolution...")
        fitness_history = runner.results["GA"]["fitness_histories"][0]
        plot_fitness_evolution(
            fitness_history,
            title=f"GA Fitness Evolution: {dataset_name.title()}",
            save_path=plots_dir / f"{dataset_name}_fitness_evolution.png",
            show=False,
        )

    # 6. GA Combined Dashboard
    if "GA" in runner.results and "chromosomes" in runner.results["GA"]:
        logger.info("  [6/6] GA combined dashboard...")
        chromosomes = runner.results["GA"]["chromosomes"]
        frequencies = feature_selection_frequency(chromosomes)
        X, _, _, _, feature_names = load_dataset(dataset_name, split=True, random_state=42)
        fitness_history = runner.results["GA"]["fitness_histories"][0]
        diversity_history = [0.5] * len(fitness_history)  # Placeholder

        plot_combined_dashboard(
            fitness_history,
            diversity_history,
            frequencies,
            feature_names=feature_names,
            title=f"GA Dashboard: {dataset_name.title()}",
            save_path=plots_dir / f"{dataset_name}_dashboard.png",
            show=False,
        )

    plot_count = len(list(plots_dir.glob('*.png')))
    logger.info(f"  ✓ Generated {plot_count} plots in {plots_dir}")


def run_experiment(config, generate_plots=True):
    """Run complete experiment with given configuration.

    Args:
        config: Configuration dictionary
        generate_plots: Whether to generate visualizations
    """
    results_dir = Path("results") / ("quick_test" if config == QUICK_CONFIG else "full_analysis")

    logger.info("=" * 80)
    logger.info("FEATURE SELECTION EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Datasets: {', '.join(config['datasets'])}")
    logger.info(f"Model: {config['model_type'].upper()}")
    logger.info(f"Runs per method: {config['n_runs']}")
    logger.info(f"Output: {results_dir}")
    logger.info("=" * 80)

    for dataset in config["datasets"]:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"DATASET: {dataset.upper()}")
        logger.info(f"{'=' * 80}")

        # Load dataset info
        X, _, _, _, feature_names = load_dataset(dataset, split=True, random_state=config["random_state"])
        logger.info(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")

        # Initialize experiment runner
        runner = ExperimentRunner(
            dataset_name=dataset,
            model_type=config["model_type"],
            n_runs=config["n_runs"],
            random_state=config["random_state"],
            results_dir=results_dir / dataset,
        )

        # Run GA
        logger.info("Running Genetic Algorithm...")
        runner.run_ga_experiment(
            population_size=config["population_size"],
            num_generations=config["num_generations"],
            mutation_rate=config["mutation_rate"],
            early_stopping_patience=config["early_stopping_patience"],
            verbose=True,
        )

        # Run baselines
        logger.info("Running baseline methods...")
        for method in ["rfe", "lasso", "mi"]:
            runner.run_baseline_experiment(method, verbose=True)
        runner.run_all_features_baseline(verbose=True)

        # Statistical comparison
        logger.info(f"\n{'=' * 80}")
        logger.info("STATISTICAL COMPARISON")
        logger.info(f"{'=' * 80}")
        runner.compare_methods(verbose=True)

        # Generate summary
        logger.info(f"\n{runner.generate_summary_report()}")

        # Save results
        runner.save_results(f"{dataset}_results.pkl")

        # Generate visualizations
        if generate_plots:
            generate_visualizations(
                runner,
                dataset,
                X.shape[1],
                runner.results_dir / "plots"
            )

    # Final summary
    logger.info(f"\n{'=' * 80}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'=' * 80}")
    logger.info(f"Results saved to: {results_dir}")

    if generate_plots:
        logger.info("\nGenerated plots per dataset:")
        logger.info("  - {dataset}_accuracy_comparison.png")
        logger.info("  - {dataset}_feature_count_comparison.png")
        logger.info("  - {dataset}_multi_metric.png")
        logger.info("  - {dataset}_pareto.png")
        logger.info("  - {dataset}_fitness_evolution.png")
        logger.info("  - {dataset}_dashboard.png")


def main():
    """Parse arguments and run experiment."""
    parser = argparse.ArgumentParser(
        description="Run feature selection experiments with genetic algorithms"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test mode (fewer runs, single dataset)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to run (e.g., iris wine breast_cancer)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip visualization generation"
    )
    parser.add_argument(
        "--runs",
        type=int,
        help="Number of runs per method (overrides default)"
    )

    args = parser.parse_args()

    # Build configuration
    if args.quick:
        config = QUICK_CONFIG.copy()
    else:
        config = DEFAULT_CONFIG.copy()

    if args.datasets:
        config["datasets"] = args.datasets
    if args.runs:
        config["n_runs"] = args.runs

    # Run experiment
    run_experiment(config, generate_plots=not args.no_plots)


if __name__ == "__main__":
    main()
