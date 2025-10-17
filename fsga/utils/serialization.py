"""Serialization utilities for saving and loading GA results.

Supports multiple formats (pickle, JSON, numpy) and handles
population states, results, and experiment metadata.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


class SerializationError(Exception):
    """Raised when serialization/deserialization fails."""

    pass


class ResultsSerializer:
    """Serializer for GA experiment results.

    Handles saving/loading of:
    - GA results (best chromosome, fitness history, etc.)
    - Population snapshots
    - Experiment metadata
    - Numpy arrays (chromosomes, fitness arrays)

    Supports formats: pickle (.pkl), JSON (.json), numpy (.npz)

    Example:
        >>> results = ga.evolve()
        >>> serializer = ResultsSerializer()
        >>> serializer.save_results(results, 'outputs/exp1_results.pkl')
        >>>
        >>> # Later...
        >>> loaded = serializer.load_results('outputs/exp1_results.pkl')
    """

    SUPPORTED_FORMATS = ["pkl", "json", "npz"]

    def save_results(
        self,
        results: dict,
        filepath: str | Path,
        metadata: Optional[dict] = None,
        format: Optional[str] = None,
    ) -> None:
        """Save GA results to file.

        Args:
            results: Results dictionary from GeneticAlgorithm.evolve()
            filepath: Output file path
            metadata: Optional metadata (experiment name, date, config, etc.)
            format: File format ('pkl', 'json', 'npz'). Auto-detected from extension.

        Raises:
            SerializationError: If format unsupported or save fails

        Example:
            >>> results = ga.evolve()
            >>> serializer.save_results(
            ...     results,
            ...     'outputs/exp1.pkl',
            ...     metadata={'experiment': 'baseline', 'dataset': 'iris'}
            ... )
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Auto-detect format from extension
        if format is None:
            format = filepath.suffix.lstrip(".")

        if format not in self.SUPPORTED_FORMATS:
            raise SerializationError(
                f"Unsupported format: {format}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Add metadata
        save_data = {
            "results": results,
            "metadata": metadata or {},
            "saved_at": datetime.now().isoformat(),
            "version": "0.1.0",
        }

        try:
            if format == "pkl":
                self._save_pickle(save_data, filepath)
            elif format == "json":
                self._save_json(save_data, filepath)
            elif format == "npz":
                self._save_numpy(save_data, filepath)
        except Exception as e:
            raise SerializationError(f"Failed to save results: {e}")

    def load_results(self, filepath: str | Path, format: Optional[str] = None) -> dict:
        """Load GA results from file.

        Args:
            filepath: Path to saved results
            format: File format. Auto-detected from extension if not specified.

        Returns:
            dict: Dictionary with 'results' and 'metadata' keys

        Raises:
            FileNotFoundError: If file doesn't exist
            SerializationError: If format unsupported or load fails

        Example:
            >>> loaded = serializer.load_results('outputs/exp1.pkl')
            >>> results = loaded['results']
            >>> metadata = loaded['metadata']
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        # Auto-detect format
        if format is None:
            format = filepath.suffix.lstrip(".")

        if format not in self.SUPPORTED_FORMATS:
            raise SerializationError(f"Unsupported format: {format}")

        try:
            if format == "pkl":
                return self._load_pickle(filepath)
            elif format == "json":
                return self._load_json(filepath)
            elif format == "npz":
                return self._load_numpy(filepath)
        except Exception as e:
            raise SerializationError(f"Failed to load results: {e}")

    def save_population(
        self, chromosomes: np.ndarray, fitness: np.ndarray, filepath: str | Path
    ) -> None:
        """Save population snapshot (chromosomes + fitness).

        Args:
            chromosomes: Population chromosomes (shape: pop_size × num_features)
            fitness: Fitness values (shape: pop_size,)
            filepath: Output file path (.npz format)

        Example:
            >>> serializer.save_population(
            ...     population.chromosomes,
            ...     population.fitness,
            ...     'checkpoints/gen_50.npz'
            ... )
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            filepath,
            chromosomes=chromosomes,
            fitness=fitness,
            saved_at=np.array([datetime.now().isoformat()], dtype=object),
        )

    def load_population(self, filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Load population snapshot.

        Args:
            filepath: Path to saved population (.npz)

        Returns:
            tuple: (chromosomes, fitness)

        Raises:
            FileNotFoundError: If file doesn't exist

        Example:
            >>> chromosomes, fitness = serializer.load_population('checkpoints/gen_50.npz')
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Population file not found: {filepath}")

        data = np.load(filepath, allow_pickle=True)
        return data["chromosomes"], data["fitness"]

    def save_checkpoint(
        self,
        ga_state: dict,
        filepath: str | Path,
    ) -> None:
        """Save full GA checkpoint for resuming evolution.

        Args:
            ga_state: Dictionary with GA state:
                - population: Population object
                - generation: Current generation
                - best_fitness_history: List of best fitness per generation
                - config: GA configuration
            filepath: Output checkpoint path

        Example:
            >>> checkpoint = {
            ...     'population': population,
            ...     'generation': 50,
            ...     'best_fitness_history': [0.8, 0.82, ...],
            ...     'config': config_dict
            ... }
            >>> serializer.save_checkpoint(checkpoint, 'checkpoints/gen_50.pkl')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata
        checkpoint_data = {
            **ga_state,
            "checkpoint_time": datetime.now().isoformat(),
            "version": "0.1.0",
        }

        with open(filepath, "wb") as f:
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, filepath: str | Path) -> dict:
        """Load GA checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            dict: GA state dictionary

        Raises:
            FileNotFoundError: If checkpoint doesn't exist

        Example:
            >>> state = serializer.load_checkpoint('checkpoints/gen_50.pkl')
            >>> population = state['population']
            >>> generation = state['generation']
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        with open(filepath, "rb") as f:
            return pickle.load(f)

    # Private methods for format-specific serialization

    def _save_pickle(self, data: dict, filepath: Path) -> None:
        """Save as pickle (binary, preserves all Python objects)."""
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_pickle(self, filepath: Path) -> dict:
        """Load from pickle."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def _save_json(self, data: dict, filepath: Path) -> None:
        """Save as JSON (human-readable, but converts numpy arrays to lists)."""
        # Convert numpy arrays to lists for JSON serialization
        json_data = self._convert_numpy_to_lists(data)

        with open(filepath, "w") as f:
            json.dump(json_data, f, indent=2)

    def _load_json(self, filepath: Path) -> dict:
        """Load from JSON (arrays remain as lists, not numpy arrays)."""
        with open(filepath, "r") as f:
            return json.load(f)

    def _save_numpy(self, data: dict, filepath: Path) -> None:
        """Save as compressed numpy archive (.npz).

        Only saves numpy arrays from results. Metadata saved as pickled object.
        """
        results = data["results"]
        arrays = {}

        # Extract numpy arrays
        if "best_chromosome" in results:
            arrays["best_chromosome"] = results["best_chromosome"]
        if "best_fitness_history" in results:
            arrays["best_fitness_history"] = np.array(results["best_fitness_history"])
        if "avg_fitness_history" in results:
            arrays["avg_fitness_history"] = np.array(results["avg_fitness_history"])
        if "diversity_history" in results:
            arrays["diversity_history"] = np.array(results["diversity_history"])

        # Save metadata separately
        arrays["metadata"] = np.array([data["metadata"]], dtype=object)
        arrays["saved_at"] = np.array([data["saved_at"]], dtype=object)

        # Save scalar values
        if "best_fitness" in results:
            arrays["best_fitness"] = np.array(results["best_fitness"])
        if "num_features_selected" in results:
            arrays["num_features_selected"] = np.array(results["num_features_selected"])

        np.savez_compressed(filepath, **arrays)

    def _load_numpy(self, filepath: Path) -> dict:
        """Load from numpy archive."""
        data = np.load(filepath, allow_pickle=True)

        results = {}
        if "best_chromosome" in data:
            results["best_chromosome"] = data["best_chromosome"]
        if "best_fitness" in data:
            results["best_fitness"] = float(data["best_fitness"])
        if "best_fitness_history" in data:
            results["best_fitness_history"] = data["best_fitness_history"].tolist()
        if "avg_fitness_history" in data:
            results["avg_fitness_history"] = data["avg_fitness_history"].tolist()
        if "diversity_history" in data:
            results["diversity_history"] = data["diversity_history"].tolist()
        if "num_features_selected" in data:
            results["num_features_selected"] = int(data["num_features_selected"])

        metadata = data["metadata"].item() if "metadata" in data else {}
        saved_at = data["saved_at"].item() if "saved_at" in data else None

        return {"results": results, "metadata": metadata, "saved_at": saved_at}

    def _convert_numpy_to_lists(self, obj: Any) -> Any:
        """Recursively convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_to_lists(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
