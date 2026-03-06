"""
Training metrics tracking utilities.

Provides a MetricsTracker class for aggregating, logging, and plotting
training metrics such as loss, accuracy, and perplexity.
"""

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks and aggregates metrics during training and evaluation.

    Maintains running averages for each metric and provides utilities
    for logging and serialization.

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.update("train_loss", 2.3)
        >>> tracker.update("train_loss", 2.1)
        >>> print(tracker.average("train_loss"))  # 2.2
    """

    def __init__(self):
        self._metrics: Dict[str, List[float]] = defaultdict(list)

    def update(self, name: str, value: float) -> None:
        """Record a single metric value.

        Args:
            name: Metric name.
            value: Metric value.
        """
        self._metrics[name].append(float(value))

    def update_dict(self, metrics: Dict[str, float]) -> None:
        """Record multiple metric values at once.

        Args:
            metrics: Dictionary of metric names to values.
        """
        for name, value in metrics.items():
            self.update(name, value)

    def average(self, name: str, last_n: Optional[int] = None) -> float:
        """Compute the average of a metric.

        Args:
            name: Metric name.
            last_n: If provided, only average the last N values.

        Returns:
            Average metric value, or 0.0 if no values recorded.
        """
        values = self._metrics.get(name, [])
        if not values:
            return 0.0
        if last_n is not None:
            values = values[-last_n:]
        return sum(values) / len(values)

    def latest(self, name: str) -> Optional[float]:
        """Get the most recent value of a metric.

        Args:
            name: Metric name.

        Returns:
            Most recent value, or None if no values recorded.
        """
        values = self._metrics.get(name, [])
        return values[-1] if values else None

    def all_values(self, name: str) -> List[float]:
        """Get all recorded values for a metric.

        Args:
            name: Metric name.

        Returns:
            List of all recorded values.
        """
        return list(self._metrics.get(name, []))

    def reset(self, name: Optional[str] = None) -> None:
        """Reset metric history.

        Args:
            name: If provided, only reset this metric. Otherwise reset all.
        """
        if name:
            self._metrics[name] = []
        else:
            self._metrics.clear()

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get a summary of all tracked metrics.

        Returns:
            Dictionary mapping metric names to statistics dicts
            (mean, min, max, last).
        """
        summary = {}
        for name, values in self._metrics.items():
            if values:
                summary[name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "last": values[-1],
                    "count": len(values),
                }
        return summary

    def log_summary(self) -> None:
        """Log a summary of all tracked metrics."""
        for name, stats in self.summary().items():
            logger.info(
                "%s: mean=%.4f, min=%.4f, max=%.4f, last=%.4f (n=%d)",
                name,
                stats["mean"],
                stats["min"],
                stats["max"],
                stats["last"],
                stats["count"],
            )

    def plot(self, metric_names: Optional[List[str]] = None, save_path: Optional[str] = None):
        """Plot training curves for the specified metrics.

        Args:
            metric_names: List of metrics to plot. If None, plots all.
            save_path: If provided, saves the plot to this path.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available. Install it to plot metrics.")
            return

        names = metric_names or list(self._metrics.keys())
        if not names:
            logger.warning("No metrics to plot.")
            return

        fig, axes = plt.subplots(len(names), 1, figsize=(10, 4 * len(names)), squeeze=False)

        for ax, name in zip(axes[:, 0], names):
            values = self.all_values(name)
            if values:
                ax.plot(values, label=name)
                ax.set_xlabel("Step")
                ax.set_ylabel(name)
                ax.set_title(f"Training Curve: {name}")
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Training curves saved to %s", save_path)
        else:
            plt.show()

        plt.close(fig)


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss.

    Args:
        loss: Cross-entropy loss value.

    Returns:
        Perplexity value.
    """
    return math.exp(min(loss, 20.0))


def compute_accuracy(predictions: Any, labels: Any) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Predicted class indices or logits.
        labels: Ground truth labels.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    import numpy as np

    preds = np.asarray(predictions)
    lbls = np.asarray(labels)

    if preds.ndim > 1:
        preds = np.argmax(preds, axis=-1)

    return float(np.mean(preds == lbls))


def compute_f1(
    predictions: Any,
    labels: Any,
    average: str = "macro",
) -> float:
    """Compute F1 score.

    Args:
        predictions: Predicted class indices or logits.
        labels: Ground truth labels.
        average: Averaging strategy ('macro', 'micro', 'weighted').

    Returns:
        F1 score.
    """
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        logger.warning("scikit-learn not available. Cannot compute F1 score.")
        return 0.0

    import numpy as np

    preds = np.asarray(predictions)
    lbls = np.asarray(labels)

    if preds.ndim > 1:
        preds = np.argmax(preds, axis=-1)

    return float(f1_score(lbls, preds, average=average))
