"""Structured logging utilities for GA experiments.

Provides experiment loggers with file/console output, JSON formatting,
and progress tracking for long-running experiments.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class ExperimentLogger:
    """Structured logger for GA experiments.

    Logs to both console and file with structured JSON output.
    Tracks experiment metadata, generation progress, and results.

    Example:
        >>> logger = ExperimentLogger('experiments/exp1', level='INFO')
        >>> logger.log_experiment_start(config={'population_size': 50})
        >>> logger.log_generation(gen=1, best_fitness=0.85, avg_fitness=0.72)
        >>> logger.log_experiment_end(results={'best_fitness': 0.95})
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str | Path = "logs",
        level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = True,
    ):
        """Initialize experiment logger.

        Args:
            experiment_name: Name of experiment (used for log filename)
            log_dir: Directory for log files
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file

        Example:
            >>> logger = ExperimentLogger('baseline_experiment', log_dir='outputs/logs')
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(f"fsga.{experiment_name}")
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers = []  # Clear existing handlers

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            # Also create JSON log file
            self.json_log_file = self.log_dir / f"{experiment_name}_{timestamp}.jsonl"

        self.start_time: Optional[datetime] = None

    def log_experiment_start(self, config: dict) -> None:
        """Log experiment start with configuration.

        Args:
            config: Experiment configuration dictionary

        Example:
            >>> logger.log_experiment_start({
            ...     'population_size': 50,
            ...     'num_generations': 100,
            ...     'dataset': 'iris'
            ... })
        """
        self.start_time = datetime.now()

        log_entry = {
            "event": "experiment_start",
            "timestamp": self.start_time.isoformat(),
            "experiment": self.experiment_name,
            "config": config,
        }

        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Config: {json.dumps(config, indent=2)}")
        self._write_json_log(log_entry)

    def log_generation(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        diversity: Optional[float] = None,
        num_features: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Log generation progress.

        Args:
            generation: Current generation number
            best_fitness: Best fitness in population
            avg_fitness: Average fitness in population
            diversity: Optional population diversity
            num_features: Optional number of features in best solution
            **kwargs: Additional metrics to log

        Example:
            >>> logger.log_generation(
            ...     generation=10,
            ...     best_fitness=0.89,
            ...     avg_fitness=0.75,
            ...     diversity=0.42,
            ...     num_features=5
            ... )
        """
        log_entry = {
            "event": "generation",
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
        }

        if diversity is not None:
            log_entry["diversity"] = diversity
        if num_features is not None:
            log_entry["num_features"] = num_features

        # Add any additional metrics
        log_entry.update(kwargs)

        msg = (
            f"Gen {generation}: best={best_fitness:.4f}, "
            f"avg={avg_fitness:.4f}"
        )
        if num_features is not None:
            msg += f", features={num_features}"

        self.logger.info(msg)
        self._write_json_log(log_entry)

    def log_experiment_end(self, results: dict) -> None:
        """Log experiment completion with final results.

        Args:
            results: Final results dictionary

        Example:
            >>> logger.log_experiment_end({
            ...     'best_fitness': 0.95,
            ...     'generations': 87,
            ...     'converged': True
            ... })
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds() if self.start_time else 0

        log_entry = {
            "event": "experiment_end",
            "timestamp": end_time.isoformat(),
            "experiment": self.experiment_name,
            "duration_seconds": duration,
            "results": results,
        }

        self.logger.info(f"Experiment completed: {self.experiment_name}")
        self.logger.info(f"Duration: {duration:.2f}s")
        self.logger.info(f"Best fitness: {results.get('best_fitness', 'N/A')}")
        self._write_json_log(log_entry)

    def log_early_stopping(self, generation: int, reason: str) -> None:
        """Log early stopping event.

        Args:
            generation: Generation where stopping occurred
            reason: Reason for early stopping

        Example:
            >>> logger.log_early_stopping(45, "No improvement for 10 generations")
        """
        log_entry = {
            "event": "early_stopping",
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            "reason": reason,
        }

        self.logger.info(f"Early stopping at generation {generation}: {reason}")
        self._write_json_log(log_entry)

    def log_checkpoint(self, generation: int, checkpoint_path: str) -> None:
        """Log checkpoint save event.

        Args:
            generation: Generation number
            checkpoint_path: Path to checkpoint file

        Example:
            >>> logger.log_checkpoint(50, 'checkpoints/gen_50.pkl')
        """
        log_entry = {
            "event": "checkpoint",
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            "checkpoint_path": checkpoint_path,
        }

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        self._write_json_log(log_entry)

    def log_error(self, error: Exception, context: Optional[dict] = None) -> None:
        """Log error with context.

        Args:
            error: Exception that occurred
            context: Optional context dictionary

        Example:
            >>> try:
            ...     # some code
            ... except Exception as e:
            ...     logger.log_error(e, {'generation': 42})
        """
        log_entry = {
            "event": "error",
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
        }

        self.logger.error(f"Error: {error}", exc_info=True)
        self._write_json_log(log_entry)

    def log_metric(self, name: str, value: Any, **kwargs) -> None:
        """Log custom metric.

        Args:
            name: Metric name
            value: Metric value
            **kwargs: Additional metadata

        Example:
            >>> logger.log_metric('validation_accuracy', 0.92, split='val')
        """
        log_entry = {
            "event": "metric",
            "timestamp": datetime.now().isoformat(),
            "metric_name": name,
            "metric_value": value,
            **kwargs,
        }

        self.logger.info(f"Metric: {name}={value}")
        self._write_json_log(log_entry)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def _write_json_log(self, log_entry: dict) -> None:
        """Write log entry to JSON lines file."""
        if hasattr(self, "json_log_file"):
            with open(self.json_log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    """Setup a basic logger with console and optional file output.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path

    Returns:
        logging.Logger: Configured logger

    Example:
        >>> logger = setup_logger('fsga.analysis', level='DEBUG', log_file='analysis.log')
        >>> logger.info('Analysis started')
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ProgressTracker:
    """Simple progress tracker for console output.

    Example:
        >>> tracker = ProgressTracker(total=100, desc="Evolving")
        >>> for gen in range(100):
        ...     # do work
        ...     tracker.update(gen + 1, metrics={'fitness': 0.95})
    """

    def __init__(self, total: int, desc: str = "Progress"):
        """Initialize progress tracker.

        Args:
            total: Total number of steps
            desc: Description to display
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = datetime.now()

    def update(self, current: int, metrics: Optional[dict] = None) -> None:
        """Update progress.

        Args:
            current: Current step
            metrics: Optional metrics to display
        """
        self.current = current
        elapsed = (datetime.now() - self.start_time).total_seconds()
        percent = (current / self.total) * 100 if self.total > 0 else 0

        msg = f"\r{self.desc}: {current}/{self.total} ({percent:.1f}%) | {elapsed:.1f}s"

        if metrics:
            metric_str = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
            msg += f" | {metric_str}"

        print(msg, end="", flush=True)

    def finish(self) -> None:
        """Finish progress tracking."""
        print()  # New line after progress bar
