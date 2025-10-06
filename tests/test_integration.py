#!/usr/bin/env python3
"""Basic integration test: Run GA on Iris dataset.

This script tests that all core components work together end-to-end.
"""

import numpy as np

from fsga.core.genetic_algorithm import GeneticAlgorithm
from fsga.datasets.loader import load_dataset
from fsga.evaluators.accuracy_evaluator import AccuracyEvaluator
from fsga.ml.models import ModelWrapper
from fsga.operators.uniform_crossover import UniformCrossover
from fsga.mutations.bitflip_mutation import BitFlipMutation
from fsga.selectors.tournament_selector import TournamentSelector


def main():
    print("=" * 60)
    print("FSGA - Basic Integration Test")
    print("=" * 60)

    # Load Iris dataset
    print("\n1. Loading Iris dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_dataset(
        "iris", split=True, test_size=0.3, random_state=42
    )
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Features: {feature_names}")

    # Create model
    print("\n2. Creating RandomForest model...")
    model = ModelWrapper("rf", n_estimators=50, random_state=42)

    # Create evaluator
    print("3. Creating AccuracyEvaluator...")
    evaluator = AccuracyEvaluator(X_train, y_train, X_test, y_test, model)

    # Create genetic operators
    print("4. Creating genetic operators...")
    crossover = UniformCrossover()
    mutation = BitFlipMutation(probability=0.1)
    selector = TournamentSelector(evaluator, tournament_size=3)

    # Create GA
    print("5. Initializing Genetic Algorithm...")
    ga = GeneticAlgorithm(
        num_features=X_train.shape[1],  # 4 features for Iris
        evaluator=evaluator,
        selector=selector,
        crossover_operator=crossover,
        mutation_operator=mutation,
        population_size=20,
        num_generations=10,
        mutation_rate=0.1,
        early_stopping_patience=5,
        verbose=True,
    )

    # Run evolution
    print("\n6. Running evolution...")
    print("-" * 60)
    results = ga.evolve()
    print("-" * 60)

    # Display results
    print("\n7. Results:")
    print(f"   Best Fitness: {results['best_fitness']:.4f}")
    print(f"   Best Chromosome: {results['best_chromosome']}")
    print(f"   Features Selected: {results['best_chromosome'].sum()}/{len(results['best_chromosome'])}")
    print(f"   Optimal Generation: {results['optimal_generation']}")
    print(f"   Execution Time: {results['execution_time_ms']:.2f} ms")
    print(f"   Converged: {results['converged']}")

    # Show which features were selected
    selected_indices = np.where(results["best_chromosome"] == 1)[0]
    selected_features = [feature_names[i] for i in selected_indices]
    print(f"   Selected Features: {selected_features}")

    # Baseline: test accuracy with all features
    print("\n8. Baseline (all features):")
    model_baseline = ModelWrapper("rf", n_estimators=50, random_state=42)
    model_baseline.fit(X_train, y_train)
    baseline_accuracy = model_baseline.score(X_test, y_test)
    print(f"   Accuracy with all features: {baseline_accuracy:.4f}")

    # Comparison
    print("\n9. Comparison:")
    improvement = results["best_fitness"] - baseline_accuracy
    print(f"   GA Accuracy: {results['best_fitness']:.4f}")
    print(f"   Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"   Improvement: {improvement:+.4f}")
    print(f"   Features Reduced: {len(feature_names) - results['best_chromosome'].sum()}/{len(feature_names)}")

    print("\n" + "=" * 60)
    print("✅ Integration test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
