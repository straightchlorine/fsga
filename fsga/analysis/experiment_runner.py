"""Experiment runner for systematic GA evaluation and comparison.

Provides a unified framework for running experiments, comparing methods,
and generating results with statistical analysis.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

from fsga.analysis.baselines import get_baseline_selector
from fsga.core.genetic_algorithm import GeneticAlgorithm
from fsga.datasets.loader import load_dataset
from fsga.evaluators.accuracy_evaluator import AccuracyEvaluator
from fsga.ml.models import ModelWrapper
from fsga.mutations.bitflip_mutation import BitFlipMutation
from fsga.operators.uniform_crossover import UniformCrossover
from fsga.selectors.tournament_selector import TournamentSelector
from fsga.utils.metrics import (
    cohens_d,
    jaccard_stability,
    mann_whitney_test,
    wilcoxon_test,
)
from fsga.utils.serialization import ResultsSerializer


class ExperimentRunner:
    """Run and compare feature selection experiments.

    Orchestrates multiple runs of GA and baseline methods,
    collects metrics, and performs statistical analysis.
    """

    def __init__(
        self,
        dataset_name: str = "iris",
        model_type: str = "rf",
        n_runs: int = 10,
        random_state: int = 42,
        results_dir: Optional[Path | str] = None,
    ):
        """Initialize experiment runner.

        Args:
            dataset_name: Dataset to use
            model_type: Model type ('rf', 'svm', etc.)
            n_runs: Number of independent runs for stability
            random_state: Base random seed
            results_dir: Directory to save results (optional)
        """
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.n_runs = n_runs
        self.random_state = random_state
        self.results_dir = Path(results_dir) if results_dir else None

        self.results = {}
        self.serializer = ResultsSerializer()

    def run_ga_experiment(
        self,
        population_size: int = 50,
        num_generations: int = 100,
        mutation_rate: float = 0.01,
        early_stopping_patience: int = 10,
        verbose: bool = False,
    ) -> dict:
        """Run multiple GA experiments.

        Args:
            population_size: GA population size
            num_generations: Maximum generations
            mutation_rate: Mutation probability
            early_stopping_patience: Early stopping patience
            verbose: Whether to print progress

        Returns:
            dict: Aggregated results from all runs
        """
        if verbose:
            print(f"Running GA experiment on {self.dataset_name} ({self.n_runs} runs)")

        all_chromosomes = []
        all_accuracies = []
        all_f1_scores = []
        all_n_features = []
        all_runtimes = []
        all_fitness_histories = []

        for run_idx in range(self.n_runs):
            seed = self.random_state + run_idx

            # Load data
            X_train, X_test, y_train, y_test, feature_names = load_dataset(
                self.dataset_name, split=True, random_state=seed
            )

            # Setup GA components
            model = ModelWrapper(self.model_type, random_state=seed)
            evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)
            selector = TournamentSelector(evaluator, tournament_size=3)
            crossover = UniformCrossover()
            mutation = BitFlipMutation(probability=mutation_rate)

            ga = GeneticAlgorithm(
                num_features=X_train.shape[1],
                evaluator=evaluator,
                selector=selector,
                crossover_operator=crossover,
                mutation_operator=mutation,
                population_size=population_size,
                num_generations=num_generations,
                early_stopping_patience=early_stopping_patience,
                verbose=False,
            )

            # Run GA
            start_time = time.time()
            results = ga.evolve()
            runtime = time.time() - start_time

            # Extract results
            best_chromosome = results["best_chromosome"]
            selected_indices = np.where(best_chromosome == 1)[0]

            # Evaluate on test set
            if len(selected_indices) > 0:
                X_train_selected = X_train[:, selected_indices]
                X_test_selected = X_test[:, selected_indices]

                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)

                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
            else:
                accuracy = 0.0
                f1 = 0.0

            all_chromosomes.append(best_chromosome)
            all_accuracies.append(accuracy)
            all_f1_scores.append(f1)
            all_n_features.append(len(selected_indices))
            all_runtimes.append(runtime)
            all_fitness_histories.append(results["best_fitness_history"])

            if verbose and (run_idx + 1) % max(1, self.n_runs // 10) == 0:
                print(
                    f"  Run {run_idx + 1}/{self.n_runs}: "
                    f"Acc={accuracy:.4f}, Features={len(selected_indices)}/{X_train.shape[1]}"
                )

        # Aggregate results
        ga_results = {
            "method": "GA",
            "chromosomes": all_chromosomes,
            "accuracies": np.array(all_accuracies),
            "f1_scores": np.array(all_f1_scores),
            "n_features": np.array(all_n_features),
            "runtimes": np.array(all_runtimes),
            "fitness_histories": all_fitness_histories,
            "mean_accuracy": np.mean(all_accuracies),
            "std_accuracy": np.std(all_accuracies),
            "mean_f1": np.mean(all_f1_scores),
            "mean_n_features": np.mean(all_n_features),
            "mean_runtime": np.mean(all_runtimes),
            "stability": jaccard_stability(all_chromosomes),
        }

        self.results["GA"] = ga_results
        return ga_results

    def run_baseline_experiment(
        self, method: str = "rfe", k: Optional[int] = None, verbose: bool = False
    ) -> dict:
        """Run baseline feature selection method.

        Args:
            method: Baseline method ('rfe', 'lasso', 'mi', 'chi2', 'anova')
            k: Number of features to select (if None, uses method default)
            verbose: Whether to print progress

        Returns:
            dict: Results from baseline method
        """
        if verbose:
            print(
                f"Running {method.upper()} baseline on {self.dataset_name} ({self.n_runs} runs)"
            )

        all_chromosomes = []
        all_accuracies = []
        all_f1_scores = []
        all_n_features = []
        all_runtimes = []

        for run_idx in range(self.n_runs):
            seed = self.random_state + run_idx

            # Load data
            X_train, X_test, y_train, y_test, feature_names = load_dataset(
                self.dataset_name, split=True, random_state=seed
            )

            # Determine k if not provided
            if k is None:
                k_to_use = X_train.shape[1] // 2  # Select half by default
            else:
                k_to_use = min(k, X_train.shape[1])

            # Run baseline
            start_time = time.time()
            selector = get_baseline_selector(method, k=k_to_use, random_state=seed)
            selector.fit(X_train, y_train)
            selected_indices = selector.get_selected_features()
            runtime = time.time() - start_time

            # Evaluate
            if len(selected_indices) > 0:
                X_train_selected = X_train[:, selected_indices]
                X_test_selected = X_test[:, selected_indices]

                model = ModelWrapper(self.model_type, random_state=seed)
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)

                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
            else:
                accuracy = 0.0
                f1 = 0.0

            chromosome = selector.get_chromosome(X_train.shape[1])
            all_chromosomes.append(chromosome)
            all_accuracies.append(accuracy)
            all_f1_scores.append(f1)
            all_n_features.append(len(selected_indices))
            all_runtimes.append(runtime)

            if verbose and (run_idx + 1) % max(1, self.n_runs // 10) == 0:
                print(
                    f"  Run {run_idx + 1}/{self.n_runs}: "
                    f"Acc={accuracy:.4f}, Features={len(selected_indices)}/{X_train.shape[1]}"
                )

        # Aggregate results
        baseline_results = {
            "method": method.upper(),
            "chromosomes": all_chromosomes,
            "accuracies": np.array(all_accuracies),
            "f1_scores": np.array(all_f1_scores),
            "n_features": np.array(all_n_features),
            "runtimes": np.array(all_runtimes),
            "mean_accuracy": np.mean(all_accuracies),
            "std_accuracy": np.std(all_accuracies),
            "mean_f1": np.mean(all_f1_scores),
            "mean_n_features": np.mean(all_n_features),
            "mean_runtime": np.mean(all_runtimes),
            "stability": jaccard_stability(all_chromosomes),
        }

        self.results[method.upper()] = baseline_results
        return baseline_results

    def run_all_features_baseline(self, verbose: bool = False) -> dict:
        """Run experiment using all features (no selection).

        Args:
            verbose: Whether to print progress

        Returns:
            dict: Results from using all features
        """
        if verbose:
            print(
                f"Running All Features baseline on {self.dataset_name} ({self.n_runs} runs)"
            )

        all_accuracies = []
        all_f1_scores = []
        all_runtimes = []

        for run_idx in range(self.n_runs):
            seed = self.random_state + run_idx

            # Load data
            X_train, X_test, y_train, y_test, feature_names = load_dataset(
                self.dataset_name, split=True, random_state=seed
            )

            # Train on all features
            start_time = time.time()
            model = ModelWrapper(self.model_type, random_state=seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            runtime = time.time() - start_time

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            all_accuracies.append(accuracy)
            all_f1_scores.append(f1)
            all_runtimes.append(runtime)

        all_features_results = {
            "method": "All Features",
            "accuracies": np.array(all_accuracies),
            "f1_scores": np.array(all_f1_scores),
            "runtimes": np.array(all_runtimes),
            "mean_accuracy": np.mean(all_accuracies),
            "std_accuracy": np.std(all_accuracies),
            "mean_f1": np.mean(all_f1_scores),
            "mean_runtime": np.mean(all_runtimes),
            "n_features": X_train.shape[1],  # All features
        }

        self.results["All Features"] = all_features_results
        return all_features_results

    def compare_methods(self, verbose: bool = True) -> dict:
        """Compare GA vs baseline methods with statistical tests.

        Args:
            verbose: Whether to print comparison summary

        Returns:
            dict: Statistical comparison results
        """
        if "GA" not in self.results:
            raise ValueError("Must run GA experiment before comparison")

        comparisons = {}
        ga_accuracies = self.results["GA"]["accuracies"]

        for method_name, method_results in self.results.items():
            if method_name == "GA":
                continue

            baseline_accuracies = method_results["accuracies"]

            # Statistical tests
            if len(ga_accuracies) == len(baseline_accuracies):
                # Paired test
                stat_test = wilcoxon_test(
                    ga_accuracies, baseline_accuracies, alternative="greater"
                )
            else:
                # Unpaired test
                stat_test = mann_whitney_test(
                    ga_accuracies, baseline_accuracies, alternative="greater"
                )

            # Effect size
            effect_size = cohens_d(ga_accuracies, baseline_accuracies)

            comparisons[method_name] = {
                "p_value": stat_test["p_value"],
                "significant": stat_test["significant"],
                "effect_size": effect_size,
                "ga_mean": np.mean(ga_accuracies),
                "baseline_mean": np.mean(baseline_accuracies),
                "improvement": np.mean(ga_accuracies) - np.mean(baseline_accuracies),
            }

            if verbose:
                print(f"\nGA vs {method_name}:")
                print(f"  GA Mean: {comparisons[method_name]['ga_mean']:.4f}")
                print(
                    f"  {method_name} Mean: {comparisons[method_name]['baseline_mean']:.4f}"
                )
                print(
                    f"  Improvement: {comparisons[method_name]['improvement']:.4f} ({comparisons[method_name]['improvement']*100:.2f}%)"
                )
                print(f"  p-value: {comparisons[method_name]['p_value']:.4f}")
                print(
                    f"  Significant: {'Yes' if comparisons[method_name]['significant'] else 'No'}"
                )
                print(f"  Effect Size (Cohen's d): {comparisons[method_name]['effect_size']:.3f}")

        return comparisons

    def save_results(self, filename: str = "experiment_results.pkl"):
        """Save all experiment results to file.

        Args:
            filename: Output filename
        """
        if self.results_dir:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.results_dir / filename
        else:
            filepath = Path(filename)

        self.serializer.save_results(
            self.results,
            filepath,
            metadata={
                "dataset": self.dataset_name,
                "model": self.model_type,
                "n_runs": self.n_runs,
                "random_state": self.random_state,
            },
        )

        print(f"Results saved to {filepath}")

    def generate_summary_report(self) -> str:
        """Generate a text summary of all results.

        Returns:
            str: Formatted summary report
        """
        report = []
        report.append("=" * 80)
        report.append(f"EXPERIMENT SUMMARY: {self.dataset_name.upper()}")
        report.append("=" * 80)
        report.append(f"Model: {self.model_type.upper()}")
        report.append(f"Runs per method: {self.n_runs}")
        report.append(f"Random seed: {self.random_state}")
        report.append("")

        # Results table
        report.append("RESULTS:")
        report.append("-" * 80)
        report.append(
            f"{'Method':<15} {'Accuracy':<12} {'F1-Score':<12} {'Features':<10} {'Runtime(s)':<12} {'Stability':<10}"
        )
        report.append("-" * 80)

        for method_name, results in self.results.items():
            acc = f"{results['mean_accuracy']:.4f} ± {results.get('std_accuracy', 0):.4f}"
            f1 = f"{results['mean_f1']:.4f}"
            n_feat = f"{results.get('mean_n_features', results.get('n_features', 'N/A')):.1f}"
            runtime = f"{results['mean_runtime']:.3f}"
            stability = f"{results.get('stability', 'N/A'):.3f}" if results.get('stability') else "N/A"

            report.append(
                f"{method_name:<15} {acc:<12} {f1:<12} {n_feat:<10} {runtime:<12} {stability:<10}"
            )

        report.append("=" * 80)

        return "\n".join(report)
