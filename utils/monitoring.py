"""
Performance monitoring utilities for Movie Recommender System
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json

from utils.logger import get_logger


@dataclass
class MetricPoint:
    """Single metric data point"""

    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores performance metrics"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.logger = get_logger("metrics")
        self._lock = threading.Lock()

    def record_metric(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric value"""
        with self._lock:
            metric_point = MetricPoint(
                timestamp=datetime.now(timezone.utc), value=value, labels=labels or {}
            )
            self.metrics[name].append(metric_point)

    def get_metric_stats(self, name: str, window_minutes: int = 5) -> Dict[str, float]:
        """Get statistics for a metric over a time window"""
        with self._lock:
            if name not in self.metrics:
                return {}

            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
            recent_points = [
                point for point in self.metrics[name] if point.timestamp >= cutoff_time
            ]

            if not recent_points:
                return {}

            values = [point.value for point in recent_points]

            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "latest": values[-1] if values else 0,
            }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        with self._lock:
            return {
                name: {
                    "count": len(points),
                    "latest": points[-1].value if points else 0,
                    "latest_timestamp": (
                        points[-1].timestamp.isoformat() if points else None
                    ),
                }
                for name, points in self.metrics.items()
            }


class PerformanceMonitor:
    """Main performance monitoring class"""

    def __init__(self):
        self.metrics = MetricsCollector()
        self.logger = get_logger("performance")
        self._monitoring_thread = None
        self._stop_monitoring = False

    def start_system_monitoring(self, interval_seconds: int = 60):
        """Start background system resource monitoring"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return

        self._stop_monitoring = False
        self._monitoring_thread = threading.Thread(
            target=self._monitor_system_resources, args=(interval_seconds,), daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("System monitoring started")

    def stop_system_monitoring(self):
        """Stop background system monitoring"""
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        self.logger.info("System monitoring stopped")

    def _monitor_system_resources(self, interval_seconds: int):
        """Monitor system resources in background"""
        while not self._stop_monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics.record_metric("system.cpu_percent", cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics.record_metric("system.memory_percent", memory.percent)
                self.metrics.record_metric(
                    "system.memory_available_gb", memory.available / (1024**3)
                )

                # Disk usage
                disk = psutil.disk_usage("/")
                self.metrics.record_metric("system.disk_percent", disk.percent)

                # Network I/O
                network = psutil.net_io_counters()
                self.metrics.record_metric(
                    "system.network_bytes_sent", network.bytes_sent
                )
                self.metrics.record_metric(
                    "system.network_bytes_recv", network.bytes_recv
                )

                time.sleep(interval_seconds)

            except Exception as e:
                self.logger.error(f"Error monitoring system resources: {e}")
                time.sleep(interval_seconds)

    def record_recommendation_request(
        self,
        user_id: int,
        algorithm: str,
        num_recommendations: int,
        response_time: float,
    ):
        """Record recommendation request metrics"""
        self.metrics.record_metric(
            "recommendations.response_time_ms",
            response_time * 1000,
            {"algorithm": algorithm, "user_id": str(user_id)},
        )

        self.metrics.record_metric("recommendations.count", 1, {"algorithm": algorithm})

        self.metrics.record_metric(
            "recommendations.num_results", num_recommendations, {"algorithm": algorithm}
        )

        self.logger.log_recommendation(
            user_id, algorithm, num_recommendations, response_time
        )

    def record_llm_query(
        self,
        model: str,
        query: str,
        response_time: float,
        num_results: int,
        success: bool,
    ):
        """Record LLM query metrics"""
        self.metrics.record_metric(
            "llm.response_time_ms",
            response_time * 1000,
            {"model": model, "success": str(success)},
        )

        self.metrics.record_metric(
            "llm.queries_count", 1, {"model": model, "success": str(success)}
        )

        if success:
            self.metrics.record_metric(
                "llm.results_count", num_results, {"model": model}
            )

        self.logger.log_llm_query(model, query, response_time, num_results, success)

    def record_user_action(
        self, user_id: int, action: str, details: Optional[Dict] = None
    ):
        """Record user action metrics"""
        labels = {
            "action": action,
            "user_id": str(user_id),
        }
        if details:
            if "ab_group" in details:
                labels["ab_group"] = details["ab_group"]
            if "algorithm" in details:
                labels["algorithm"] = details["algorithm"]
        self.metrics.record_metric("user.actions_count", 1, labels)
        self.logger.log_user_action(user_id, action, details)

    def record_model_performance(
        self, metric: str, value: float, algorithm: str = "svd"
    ):
        """Record model performance metrics"""
        self.metrics.record_metric(f"model.{metric}", value, {"algorithm": algorithm})

        self.logger.log_model_performance(metric, value, algorithm)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of current performance metrics"""
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": {},
            "recommendations": {},
            "llm": {},
            "user_actions": {},
            "model_performance": {},
        }

        # System metrics (last 5 minutes)
        for metric in [
            "system.cpu_percent",
            "system.memory_percent",
            "system.disk_percent",
        ]:
            stats = self.metrics.get_metric_stats(metric, 5)
            if stats:
                summary["system"][metric] = stats

        # Recommendation metrics (last 5 minutes)
        for metric in ["recommendations.response_time_ms", "recommendations.count"]:
            stats = self.metrics.get_metric_stats(metric, 5)
            if stats:
                summary["recommendations"][metric] = stats

        # LLM metrics (last 5 minutes)
        for metric in ["llm.response_time_ms", "llm.queries_count"]:
            stats = self.metrics.get_metric_stats(metric, 5)
            if stats:
                summary["llm"][metric] = stats

        # --- User Actions Aggregation by (action, ab_group) ---
        user_action_points = self.metrics.metrics.get("user.actions_count", [])
        action_agg = {}

        # First pass: collect all actions with their ab_group
        for point in user_action_points:
            action = point.labels.get("action", "unknown")
            ab_group = point.labels.get("ab_group", "Unknown")
            user_id = point.labels.get("user_id", "unknown")

            # Use string key for JSON serialization
            key = f"{action}_{ab_group}"

            if key not in action_agg:
                action_agg[key] = {
                    "count": 0,
                    "last_timestamp": None,
                    "ab_group": ab_group,
                    "action": action,
                    "user_ids": set(),  # Track unique users
                }

            action_agg[key]["count"] += 1
            action_agg[key]["user_ids"].add(user_id)

            if (
                action_agg[key]["last_timestamp"] is None
                or point.timestamp > action_agg[key]["last_timestamp"]
            ):
                action_agg[key]["last_timestamp"] = point.timestamp

        # Convert sets to counts and format timestamps
        for key in action_agg:
            action_agg[key]["unique_users"] = len(action_agg[key]["user_ids"])
            del action_agg[key]["user_ids"]  # Remove set, keep only count

            if action_agg[key]["last_timestamp"]:
                action_agg[key]["last_timestamp"] = action_agg[key][
                    "last_timestamp"
                ].isoformat()

        summary["user_actions"] = action_agg

        # --- Model Performance Aggregation ---
        model_perf_agg = {}
        for metric_name, points in self.metrics.metrics.items():
            if metric_name.startswith("model."):
                metric = metric_name[6:]
                algos = {}
                for point in points:
                    algo = point.labels.get("algorithm", "unknown")
                    if algo not in algos:
                        algos[algo] = []
                    algos[algo].append(point.value)
                for algo, vals in algos.items():
                    if metric not in model_perf_agg:
                        model_perf_agg[metric] = {}
                    model_perf_agg[metric][algo] = {
                        "latest": vals[-1],
                        "min": min(vals),
                        "max": max(vals),
                        "mean": sum(vals) / len(vals),
                        "count": len(vals),
                    }
        summary["model_performance"] = model_perf_agg
        return summary

    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        try:
            with open(filepath, "w") as f:
                json.dump(self.metrics.get_all_metrics(), f, indent=2, default=str)
            self.logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")

    def clear_metrics(self):
        """Clear all stored metrics"""
        with self._lock:
            self.metrics.clear()
        self.logger.info("All metrics cleared")


# Global performance monitor instance
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


# Performance monitoring decorator
def monitor_performance(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to monitor function performance"""

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Record metric
                metric_labels = labels or {}
                metric_labels["function"] = func.__name__
                metric_labels["module"] = func.__module__

                monitor.metrics.record_metric(
                    f"function.{metric_name}",
                    execution_time * 1000,  # Convert to milliseconds
                    metric_labels,
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                # Record error metric
                metric_labels = labels or {}
                metric_labels["function"] = func.__name__
                metric_labels["module"] = func.__module__
                metric_labels["error"] = "true"

                monitor.metrics.record_metric(
                    f"function.{metric_name}", execution_time * 1000, metric_labels
                )

                raise

        return wrapper

    return decorator


# Convenience functions
def record_recommendation(
    user_id: int, algorithm: str, num_recommendations: int, response_time: float
):
    """Quick function to record recommendation metrics"""
    get_performance_monitor().record_recommendation_request(
        user_id, algorithm, num_recommendations, response_time
    )


def record_llm_query(
    model: str, query: str, response_time: float, num_results: int, success: bool
):
    """Quick function to record LLM query metrics"""
    get_performance_monitor().record_llm_query(
        model, query, response_time, num_results, success
    )


def record_user_action(user_id: int, action: str, details: Optional[Dict] = None):
    """Quick function to record user action metrics"""
    get_performance_monitor().record_user_action(user_id, action, details)


def get_performance_summary() -> Dict[str, Any]:
    """Quick function to get performance summary"""
    return get_performance_monitor().get_performance_summary()
