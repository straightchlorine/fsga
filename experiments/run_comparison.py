#!/usr/bin/env python3
"""Example experiment: Compare GA vs baseline methods on multiple datasets.

This script demonstrates how to use the ExperimentRunner to systematically
compare the GA approach against baseline feature selection methods.
"""

from pathlib import Path

from fsga.analysis.experiment_runner import ExperimentRunner
from fsga.visualization.plots import plot_method_comparison

# Configuration
DATASETS = ["iris", "wine", "breast_cancer"]
MODEL_TYPE = "rf"
N_RUNS = 10
RANDOM_STATE = 42
RESULTS_DIR = Path("results") / "comparison_study"


def main():
    """Run comparison experiment across multiple datasets."""
    print("=" * 80)
    print("FEATURE SELECTION COMPARISON STUDY")
    print("=" * 80)
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Model: {MODEL_TYPE.upper()}")
    print(f"Runs per method: {N_RUNS}")
    print("=" * 80)
    print()

    all_results = {}

    for dataset in DATASETS:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*80}\n")

        # Initialize experiment runner
        runner = ExperimentRunner(
            dataset_name=dataset,
            model_type=MODEL_TYPE,
            n_runs=N_RUNS,
            random_state=RANDOM_STATE,
            results_dir=RESULTS_DIR / dataset,
        )

        # Run GA
        print(f"Running Genetic Algorithm...")
        runner.run_ga_experiment(
            population_size=50,
            num_generations=100,
            mutation_rate=0.01,
            early_stopping_patience=10,
            verbose=True,
        )

        # Run baselines
        print(f"\nRunning baseline methods...")
        runner.run_baseline_experiment("rfe", verbose=True)
        runner.run_baseline_experiment("lasso", verbose=True)
        runner.run_baseline_experiment("mi", verbose=True)
        runner.run_all_features_baseline(verbose=True)

        # Compare methods
        print(f"\n{'='*80}")
        print("STATISTICAL COMPARISON")
        print(f"{'='*80}")
        comparisons = runner.compare_methods(verbose=True)

        # Generate summary
        print(f"\n{runner.generate_summary_report()}")

        # Save results
        runner.save_results(f"{dataset}_results.pkl")

        # Store for cross-dataset analysis
        all_results[dataset] = runner.results

        # Create comparison plot
        method_accuracies = {
            method: results["accuracies"]
            for method, results in runner.results.items()
        }

        fig = plot_method_comparison(
            method_accuracies,
            metric_name="Accuracy",
            title=f"Method Comparison: {dataset.title()}",
            save_path=RESULTS_DIR / dataset / f"{dataset}_comparison.png",
            show=False,
        )
        print(f"Saved comparison plot to {RESULTS_DIR / dataset / f'{dataset}_comparison.png'}")

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("\nTo analyze results:")
    print(f"  from fsga.utils.serialization import ResultsSerializer")
    print(f"  serializer = ResultsSerializer()")
    print(f"  results = serializer.load_results('{RESULTS_DIR}/iris/iris_results.pkl')")


if __name__ == "__main__":
    main()
