"""Tests for utils module (config, serialization, metrics, logging)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fsga.utils.config import Config, ConfigError
from fsga.utils.metrics import (
    average_sparsity,
    cohens_d,
    convergence_detected,
    core_features,
    effect_size_interpretation,
    feature_selection_frequency,
    jaccard_similarity,
    jaccard_stability,
    mann_whitney_test,
    population_diversity,
    sparsity,
    wilcoxon_test,
)
from fsga.utils.serialization import ResultsSerializer, SerializationError


class TestConfig:
    """Test configuration management."""

    def test_from_dict_basic(self):
        """Test creating config from dictionary."""
        config_dict = {
            "population_size": 50,
            "num_generations": 100,
            "mutation_rate": 0.01,
            "dataset": {"name": "iris"},
            "model": {"type": "rf"},
        }
        config = Config.from_dict(config_dict)
        assert config.population_size == 50
        assert config.num_generations == 100

    def test_get_with_dot_notation(self):
        """Test accessing nested values with dot notation."""
        config_dict = {
            "population_size": 50,
            "dataset": {"name": "iris", "test_size": 0.2},
            "model": {"type": "rf"},
        }
        config = Config.from_dict(config_dict)
        assert config.get("dataset.name") == "iris"
        assert config.get("dataset.test_size") == 0.2

    def test_get_with_default(self):
        """Test get with default value for missing keys."""
        config_dict = {
            "population_size": 50,
            "dataset": {"name": "iris"},
            "model": {"type": "rf"},
        }
        config = Config.from_dict(config_dict)
        assert config.get("nonexistent", "default") == "default"
        assert config.get("dataset.nonexistent", 123) == 123

    def test_set_value(self):
        """Test setting config values."""
        config_dict = {
            "population_size": 50,
            "dataset": {"name": "iris"},
            "model": {"type": "rf"},
        }
        config = Config.from_dict(config_dict)
        config.set("population_size", 100)
        assert config.population_size == 100

        config.set("dataset.name", "wine")
        assert config.get("dataset.name") == "wine"

    def test_merge_configs(self):
        """Test merging two configs."""
        base_dict = {
            "population_size": 50,
            "mutation_rate": 0.01,
            "dataset": {"name": "iris"},
            "model": {"type": "rf"},
        }
        override_dict = {
            "population_size": 100,
            "dataset": {"name": "wine", "test_size": 0.3},
            "model": {"type": "rf"},
        }

        base_config = Config.from_dict(base_dict)
        override_config = Config.from_dict(override_dict)
        merged = base_config.merge(override_config)

        assert merged.population_size == 100
        assert merged.mutation_rate == 0.01  # From base
        assert merged.get("dataset.name") == "wine"
        assert merged.get("dataset.test_size") == 0.3

    def test_missing_required_field_raises_error(self):
        """Test that missing required fields raise ConfigError."""
        config_dict = {"population_size": 50}  # Missing dataset.name, model.type, etc.

        with pytest.raises(ConfigError, match="Missing required config fields"):
            Config.from_dict(config_dict)

    def test_save_and_load_yaml(self):
        """Test saving and loading config from YAML."""
        config_dict = {
            "population_size": 50,
            "num_generations": 100,
            "mutation_rate": 0.01,
            "dataset": {"name": "iris"},
            "model": {"type": "rf"},
        }
        config = Config.from_dict(config_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.yaml"
            config.save(filepath)

            loaded = Config.from_file(filepath)
            assert loaded.population_size == 50
            assert loaded.get("dataset.name") == "iris"

    def test_load_nonexistent_file_raises_error(self):
        """Test loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Config.from_file("nonexistent.yaml")

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config_dict = {
            "population_size": 50,
            "dataset": {"name": "iris"},
            "model": {"type": "rf"},
        }
        config = Config.from_dict(config_dict)
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["population_size"] == 50
        assert result["dataset"]["name"] == "iris"

    def test_attribute_access(self):
        """Test accessing config values as attributes."""
        config_dict = {
            "population_size": 50,
            "num_generations": 100,
            "mutation_rate": 0.01,
            "dataset": {"name": "iris"},
            "model": {"type": "rf"},
        }
        config = Config.from_dict(config_dict)

        assert config.population_size == 50
        assert config.num_generations == 100
        assert config.mutation_rate == 0.01

    def test_invalid_attribute_raises_error(self):
        """Test accessing nonexistent attribute raises AttributeError."""
        config_dict = {
            "population_size": 50,
            "dataset": {"name": "iris"},
            "model": {"type": "rf"},
        }
        config = Config.from_dict(config_dict)

        with pytest.raises(AttributeError):
            _ = config.nonexistent_field


class TestSerialization:
    """Test serialization utilities."""

    def test_save_and_load_pickle(self):
        """Test saving and loading results as pickle."""
        results = {
            "best_chromosome": np.array([1, 0, 1, 1, 0]),
            "best_fitness": 0.95,
            "best_fitness_history": [0.8, 0.85, 0.9, 0.95],
        }

        serializer = ResultsSerializer()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.pkl"
            serializer.save_results(results, filepath, metadata={"exp": "test"})

            loaded = serializer.load_results(filepath)
            assert loaded["results"]["best_fitness"] == 0.95
            assert len(loaded["results"]["best_fitness_history"]) == 4
            assert loaded["metadata"]["exp"] == "test"

    def test_save_and_load_json(self):
        """Test saving and loading results as JSON."""
        results = {
            "best_chromosome": np.array([1, 0, 1]),
            "best_fitness": 0.95,
            "best_fitness_history": [0.8, 0.85, 0.9],
        }

        serializer = ResultsSerializer()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.json"
            serializer.save_results(results, filepath)

            loaded = serializer.load_results(filepath)
            assert loaded["results"]["best_fitness"] == 0.95
            # Note: numpy arrays converted to lists in JSON
            assert isinstance(loaded["results"]["best_chromosome"], list)

    def test_save_and_load_numpy(self):
        """Test saving and loading results as numpy archive."""
        results = {
            "best_chromosome": np.array([1, 0, 1, 1, 0]),
            "best_fitness": 0.95,
            "best_fitness_history": [0.8, 0.85, 0.9, 0.95],
            "num_features_selected": 3,
        }

        serializer = ResultsSerializer()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.npz"
            serializer.save_results(results, filepath)

            loaded = serializer.load_results(filepath)
            assert loaded["results"]["best_fitness"] == 0.95
            assert loaded["results"]["num_features_selected"] == 3
            np.testing.assert_array_equal(
                loaded["results"]["best_chromosome"], np.array([1, 0, 1, 1, 0])
            )

    def test_unsupported_format_raises_error(self):
        """Test unsupported format raises SerializationError."""
        results = {"best_fitness": 0.95}
        serializer = ResultsSerializer()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.txt"
            with pytest.raises(SerializationError, match="Unsupported format"):
                serializer.save_results(results, filepath)

    def test_save_and_load_population(self):
        """Test saving and loading population snapshot."""
        chromosomes = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        fitness = np.array([0.8, 0.9, 0.85])

        serializer = ResultsSerializer()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "population.npz"
            serializer.save_population(chromosomes, fitness, filepath)

            loaded_chroms, loaded_fitness = serializer.load_population(filepath)
            np.testing.assert_array_equal(loaded_chroms, chromosomes)
            np.testing.assert_array_equal(loaded_fitness, fitness)

    def test_save_and_load_checkpoint(self):
        """Test saving and loading full GA checkpoint."""
        checkpoint = {
            "generation": 50,
            "best_fitness_history": [0.7, 0.75, 0.8],
            "config": {"population_size": 50},
        }

        serializer = ResultsSerializer()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "checkpoint.pkl"
            serializer.save_checkpoint(checkpoint, filepath)

            loaded = serializer.load_checkpoint(filepath)
            assert loaded["generation"] == 50
            assert len(loaded["best_fitness_history"]) == 3
            assert loaded["config"]["population_size"] == 50


class TestMetrics:
    """Test metrics utilities."""

    def test_jaccard_similarity_identical(self):
        """Test Jaccard similarity for identical sets."""
        a = np.array([1, 1, 0, 0, 1])
        b = np.array([1, 1, 0, 0, 1])
        assert jaccard_similarity(a, b) == 1.0

    def test_jaccard_similarity_different(self):
        """Test Jaccard similarity for different sets."""
        a = np.array([1, 1, 0, 0, 0])
        b = np.array([0, 0, 1, 1, 1])
        assert jaccard_similarity(a, b) == 0.0

    def test_jaccard_similarity_partial(self):
        """Test Jaccard similarity for partial overlap."""
        a = np.array([1, 1, 0, 0, 1])  # {0, 1, 4}
        b = np.array([1, 0, 0, 1, 1])  # {0, 3, 4}
        # Intersection: {0, 4} = 2
        # Union: {0, 1, 3, 4} = 4
        assert jaccard_similarity(a, b) == 0.5

    def test_jaccard_similarity_both_empty(self):
        """Test Jaccard similarity when both sets are empty."""
        a = np.array([0, 0, 0])
        b = np.array([0, 0, 0])
        assert jaccard_similarity(a, b) == 1.0  # Both empty = identical

    def test_jaccard_stability(self):
        """Test Jaccard stability across multiple chromosomes."""
        chromosomes = [
            np.array([1, 1, 0, 0, 1]),
            np.array([1, 1, 0, 0, 1]),
            np.array([1, 1, 0, 0, 1]),
        ]
        # All identical -> stability = 1.0
        assert jaccard_stability(chromosomes) == 1.0

    def test_jaccard_stability_variable(self):
        """Test Jaccard stability with some variation."""
        chromosomes = [
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 1, 0]),
            np.array([0, 1, 1, 0]),
        ]
        stability = jaccard_stability(chromosomes)
        assert 0.0 < stability < 1.0

    def test_cohens_d_positive(self):
        """Test Cohen's d when group_a better than group_b."""
        group_a = np.array([0.95, 0.94, 0.96, 0.95])
        group_b = np.array([0.85, 0.84, 0.86, 0.85])
        d = cohens_d(group_a, group_b)
        assert d > 0  # Positive = group_a better

    def test_cohens_d_negative(self):
        """Test Cohen's d when group_b better than group_a."""
        group_a = np.array([0.85, 0.84, 0.86])
        group_b = np.array([0.95, 0.94, 0.96])
        d = cohens_d(group_a, group_b)
        assert d < 0  # Negative = group_b better

    def test_cohens_d_identical(self):
        """Test Cohen's d for identical groups."""
        group_a = np.array([0.9, 0.9, 0.9])
        group_b = np.array([0.9, 0.9, 0.9])
        d = cohens_d(group_a, group_b)
        assert abs(d) < 1e-6  # Should be ~0

    def test_effect_size_interpretation(self):
        """Test effect size interpretation."""
        assert effect_size_interpretation(0.1) == "negligible"
        assert effect_size_interpretation(0.3) == "small"
        assert effect_size_interpretation(0.6) == "medium"
        assert effect_size_interpretation(1.0) == "large"

    def test_convergence_detected_early(self):
        """Test convergence detection when GA converges early."""
        history = [0.7, 0.75, 0.8, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82]
        conv_gen = convergence_detected(history, patience=5, min_delta=0.001)
        assert conv_gen is not None
        assert conv_gen >= 5

    def test_convergence_not_detected(self):
        """Test no convergence when fitness keeps improving."""
        history = [0.7, 0.75, 0.8, 0.85, 0.9]
        conv_gen = convergence_detected(history, patience=3, min_delta=0.01)
        assert conv_gen is None

    def test_population_diversity_high(self):
        """Test population diversity for diverse population."""
        chromosomes = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        diversity = population_diversity(chromosomes)
        assert diversity == 0.5  # Each pair differs in exactly 2/4 positions = 0.5 diversity

    def test_population_diversity_low(self):
        """Test population diversity for converged population."""
        chromosomes = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])
        diversity = population_diversity(chromosomes)
        assert diversity == 0.0  # All identical

    def test_feature_selection_frequency(self):
        """Test feature selection frequency calculation."""
        chromosomes = [
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 1, 0]),
        ]
        frequencies = feature_selection_frequency(chromosomes)
        np.testing.assert_array_almost_equal(
            frequencies, np.array([1.0, 2 / 3, 2 / 3, 0.0])
        )

    def test_core_features(self):
        """Test core feature extraction."""
        chromosomes = [
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 1, 0]),
            np.array([1, 1, 0, 0]),
        ]
        # Feature 0: 4/4 = 100%
        # Feature 1: 3/4 = 75%
        # Feature 2: 2/4 = 50%
        # Feature 3: 0/4 = 0%

        core = core_features(chromosomes, threshold=0.8)
        assert 0 in core  # Feature 0 selected in 100% of runs

        core = core_features(chromosomes, threshold=0.5)
        assert 0 in core
        assert 1 in core
        assert 2 in core

    def test_core_features_with_names(self):
        """Test core features with feature names."""
        chromosomes = [
            np.array([1, 1, 0]),
            np.array([1, 1, 0]),
        ]
        names = ["feature_a", "feature_b", "feature_c"]
        core = core_features(chromosomes, threshold=0.8, feature_names=names)
        assert "feature_a" in core
        assert "feature_b" in core
        assert "feature_c" not in core

    def test_sparsity(self):
        """Test sparsity calculation."""
        chromosome = np.array([1, 0, 0, 1, 0])  # 2/5 selected
        assert sparsity(chromosome) == 0.6  # 3/5 not selected

    def test_sparsity_all_selected(self):
        """Test sparsity when all features selected."""
        chromosome = np.array([1, 1, 1, 1])
        assert sparsity(chromosome) == 0.0

    def test_sparsity_none_selected(self):
        """Test sparsity when no features selected."""
        chromosome = np.array([0, 0, 0, 0])
        assert sparsity(chromosome) == 1.0

    def test_average_sparsity(self):
        """Test average sparsity across chromosomes."""
        chromosomes = [
            np.array([1, 0, 0, 0]),  # sparsity = 0.75
            np.array([1, 1, 0, 0]),  # sparsity = 0.5
            np.array([1, 1, 1, 0]),  # sparsity = 0.25
        ]
        avg = average_sparsity(chromosomes)
        assert abs(avg - 0.5) < 1e-6

    def test_wilcoxon_test(self):
        """Test Wilcoxon signed-rank test."""
        # Group A clearly better than Group B
        group_a = np.array([0.95, 0.94, 0.96, 0.95, 0.94])
        group_b = np.array([0.85, 0.84, 0.86, 0.85, 0.84])

        result = wilcoxon_test(group_a, group_b, alternative="greater")
        assert "p_value" in result
        assert "statistic" in result
        assert "significant" in result
        assert result["significant"] == True  # Should be significant

    def test_wilcoxon_test_identical(self):
        """Test Wilcoxon test on identical groups."""
        group_a = np.array([0.9, 0.9, 0.9])
        group_b = np.array([0.9, 0.9, 0.9])

        result = wilcoxon_test(group_a, group_b)
        assert result["significant"] == False  # No difference

    def test_mann_whitney_test(self):
        """Test Mann-Whitney U test."""
        group_a = np.array([0.95, 0.94, 0.96])
        group_b = np.array([0.85, 0.84])

        result = mann_whitney_test(group_a, group_b, alternative="greater")
        assert "p_value" in result
        assert "statistic" in result
        assert "significant" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
