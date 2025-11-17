"""Configuration management for FSGA experiments.

Handles loading, validation, and merging of YAML configuration files
for genetic algorithm experiments.
"""

from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when configuration is invalid or missing required fields."""

    pass


class Config:
    """Configuration manager for GA experiments.

    Loads YAML configs, validates required fields, provides defaults,
    and supports config merging for inheritance.

    Example YAML config:
        ```yaml
        # config/base.yaml
        population_size: 50
        num_generations: 100
        mutation_rate: 0.01
        crossover_rate: 0.9

        dataset:
          name: iris
          test_size: 0.2
          random_state: 42

        model:
          type: rf
          n_estimators: 100
          random_state: 42

        selector:
          type: tournament
          tournament_size: 3

        operators:
          crossover: uniform
          mutation: bitflip

        evaluator:
          type: accuracy

        early_stopping:
          enabled: true
          patience: 10
          min_delta: 0.001
        ```

    Example:
        >>> config = Config.from_file('config/experiment1.yaml')
        >>> print(config.population_size)
        50
        >>> print(config.get('dataset.name'))
        'iris'
    """

    REQUIRED_FIELDS = [
        "population_size",
        "num_generations",
        "mutation_rate",
        "dataset.name",
        "model.type",
    ]

    DEFAULT_CONFIG = {
        "population_size": 50,
        "num_generations": 100,
        "mutation_rate": 0.01,
        "crossover_rate": 0.9,
        "tournament_size": 3,
        "early_stopping": {"enabled": False, "patience": 10, "min_delta": 0.001},
        "verbose": True,
        "random_state": None,
    }

    def __init__(self, config_dict: dict):
        """Initialize config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Raises:
            ConfigError: If required fields are missing
        """
        # Merge with defaults
        self._config = self._merge_configs(self.DEFAULT_CONFIG.copy(), config_dict)

        # Validate required fields
        self._validate()

    @classmethod
    def from_file(cls, filepath: str | Path) -> "Config":
        """Load configuration from YAML file.

        Args:
            filepath: Path to YAML config file

        Returns:
            Config: Initialized configuration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ConfigError: If YAML is invalid or required fields missing

        Example:
            >>> config = Config.from_file('experiments/config.yaml')
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        try:
            with open(filepath) as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {filepath}: {e}") from e

        if config_dict is None:
            config_dict = {}

        return cls(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config: Initialized configuration object

        Example:
            >>> config = Config.from_dict({
            ...     'population_size': 100,
            ...     'dataset': {'name': 'wine'}
            ... })
        """
        return cls(config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation.

        Args:
            key: Config key (supports dot notation like 'dataset.name')
            default: Default value if key not found

        Returns:
            Config value or default

        Example:
            >>> config.get('dataset.name')
            'iris'
            >>> config.get('dataset.unknown', 'default_value')
            'default_value'
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set config value using dot notation.

        Args:
            key: Config key (supports dot notation)
            value: Value to set

        Example:
            >>> config.set('population_size', 100)
            >>> config.set('dataset.name', 'wine')
        """
        keys = key.split(".")
        current = self._config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            dict: Complete configuration as nested dictionary

        Example:
            >>> config_dict = config.to_dict()
        """
        return self._config.copy()

    def save(self, filepath: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            filepath: Path to save YAML config

        Example:
            >>> config.save('experiments/saved_config.yaml')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def merge(self, other: "Config") -> "Config":
        """Merge with another config (other takes precedence).

        Args:
            other: Config to merge

        Returns:
            Config: New merged configuration

        Example:
            >>> base_config = Config.from_file('base.yaml')
            >>> exp_config = Config.from_file('experiment.yaml')
            >>> merged = base_config.merge(exp_config)
        """
        merged_dict = self._merge_configs(self._config, other._config)
        return Config(merged_dict)

    def _validate(self) -> None:
        """Validate that all required fields are present.

        Raises:
            ConfigError: If required fields are missing
        """
        missing = []

        for field in self.REQUIRED_FIELDS:
            if self.get(field) is None:
                missing.append(field)

        if missing:
            raise ConfigError(
                f"Missing required config fields: {', '.join(missing)}\n"
                f"Required fields: {', '.join(self.REQUIRED_FIELDS)}"
            )

    def _merge_configs(self, base: dict, override: dict) -> dict:
        """Recursively merge two config dictionaries.

        Args:
            base: Base configuration
            override: Override configuration (takes precedence)

        Returns:
            dict: Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def __getattr__(self, name: str) -> Any:
        """Access config values as attributes.

        Args:
            name: Config key

        Returns:
            Config value

        Raises:
            AttributeError: If key not found

        Example:
            >>> config.population_size
            50
        """
        if name.startswith("_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'"
            )

        if name in self._config:
            return self._config[name]

        raise AttributeError(f"Config has no field '{name}'")

    def __repr__(self) -> str:
        """String representation of config."""
        return f"Config({self._config})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return yaml.dump(self._config, default_flow_style=False, sort_keys=False)
