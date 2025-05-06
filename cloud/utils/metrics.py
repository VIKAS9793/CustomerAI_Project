"""
Performance metrics utilities for cloud services.

This module provides tools for tracking and reporting performance metrics
for cloud service operations.

Copyright (c) 2025 Vikas Sahani
GitHub: https://github.com/VIKAS9793
Email: vikassahani17@gmail.com

Licensed under MIT License - see LICENSE file for details
"""

import functools
import json
import logging
import statistics
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from cloud.config import CloudProvider
from src.utils.date_provider import DateProvider

logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar("T")


class MetricType(Enum):
    """Types of performance metrics collected."""

    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    SUCCESS_RATE = "success_rate"
    RETRY_COUNT = "retry_count"
    RESOURCE_USAGE = "resource_usage"


class MetricsManager:
    """
    Manager for collecting and reporting performance metrics.

    This class provides methods for tracking cloud service performance,
    calculating statistics, and generating reports.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MetricsManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize metrics manager (only once for singleton)."""
        if self._initialized:
            return

        # Metrics storage
        self.metrics = {
            provider.value: {
                service: {
                    operation: {metric_type.value: [] for metric_type in MetricType}
                    for operation in ["all"]
                }
                for service in ["all"]
            }
            for provider in CloudProvider
        }

        # Aggregated metrics (calculated periodically)
        self.aggregated = {}

        # Last aggregation time
        self.last_aggregation = time.time()

        # Configuration
        self.enabled = True
        self.aggregation_interval = 60  # seconds
        self.retention_period = 3600  # seconds
        self.max_samples = 1000  # per metric

        # Thread safety
        self.lock = threading.RLock()

        self._initialized = True

    def record_metric(
        self,
        provider: Union[CloudProvider, str],
        service: str,
        operation: str,
        metric_type: Union[MetricType, str],
        value: Any,
    ) -> None:
        """
        Record a metric value.

        Args:
            provider: Cloud provider
            service: Service name
            operation: Operation name
            metric_type: Type of metric
            value: Metric value
        """
        if not self.enabled:
            return

        # Normalize provider
        if isinstance(provider, str):
            try:
                provider_value = CloudProvider(provider.lower()).value
            except ValueError:
                provider_value = "unknown"
        else:
            provider_value = provider.value

        # Normalize metric type
        if isinstance(metric_type, str):
            try:
                metric_type_value = MetricType(metric_type.lower()).value
            except ValueError:
                metric_type_value = "unknown"
        else:
            metric_type_value = metric_type.value

        # Ensure all required structures exist
        with self.lock:
            # Create provider entry if needed
            if provider_value not in self.metrics:
                self.metrics[provider_value] = {}

            # Create service entry if needed
            if service not in self.metrics[provider_value]:
                self.metrics[provider_value][service] = {}

            # Create "all" operation entry if needed
            if "all" not in self.metrics[provider_value][service]:
                self.metrics[provider_value][service]["all"] = {
                    metric.value: [] for metric in MetricType
                }

            # Create operation entry if needed
            if operation not in self.metrics[provider_value][service]:
                self.metrics[provider_value][service][operation] = {
                    metric.value: [] for metric in MetricType
                }

            # Create metric type entry if needed
            if metric_type_value not in self.metrics[provider_value][service][operation]:
                self.metrics[provider_value][service][operation][metric_type_value] = []

            # Record timestamp with value
            current_time = time.time()

            # Add to specific operation
            metric_data = {"timestamp": current_time, "value": value}
            metric_list = self.metrics[provider_value][service][operation][metric_type_value]
            metric_list.append(metric_data)

            # Also add to "all" operations for this service
            all_op_list = self.metrics[provider_value][service]["all"][metric_type_value]
            all_op_list.append(metric_data)

            # Also add to "all" services for this provider if service is not "all"
            if service != "all" and "all" in self.metrics[provider_value]:
                all_svc_list = self.metrics[provider_value]["all"]["all"][metric_type_value]
                all_svc_list.append(metric_data)

            # Prune old metrics or if max samples exceeded
            self._prune_metrics()

            # Check if we need to aggregate
            if current_time - self.last_aggregation >= self.aggregation_interval:
                self._aggregate_metrics()

    def _prune_metrics(self) -> None:
        """Remove old metrics or limit number of samples."""
        current_time = time.time()
        cutoff_time = current_time - self.retention_period

        for provider in self.metrics:
            for service in self.metrics[provider]:
                for operation in self.metrics[provider][service]:
                    for metric_type in self.metrics[provider][service][operation]:
                        metric_list = self.metrics[provider][service][operation][metric_type]

                        # Remove old metrics
                        metric_list = [m for m in metric_list if m["timestamp"] >= cutoff_time]

                        # Limit number of samples if needed
                        if len(metric_list) > self.max_samples:
                            metric_list = metric_list[-self.max_samples :]

                        self.metrics[provider][service][operation][metric_type] = metric_list

    def _aggregate_metrics(self) -> None:
        """Aggregate metrics for reporting."""
        with self.lock:
            aggregated = {}
            current_time = time.time()
            self.last_aggregation = current_time

            for provider in self.metrics:
                aggregated[provider] = {}

                for service in self.metrics[provider]:
                    aggregated[provider][service] = {}

                    for operation in self.metrics[provider][service]:
                        aggregated[provider][service][operation] = {}

                        for metric_type in self.metrics[provider][service][operation]:
                            metric_list = self.metrics[provider][service][operation][metric_type]

                            # Skip if no data
                            if not metric_list:
                                continue

                            # Get values
                            values = [m["value"] for m in metric_list]

                            # Calculate statistics
                            stats = {
                                "count": len(values),
                                "last_value": values[-1] if values else None,
                            }

                            # Numeric statistics if possible
                            numeric_values = [v for v in values if isinstance(v, (int, float))]
                            if numeric_values:
                                stats.update(
                                    {
                                        "min": min(numeric_values),
                                        "max": max(numeric_values),
                                        "mean": statistics.mean(numeric_values),
                                        "median": statistics.median(numeric_values),
                                    }
                                )

                                # Calculate percentiles if enough data
                                if len(numeric_values) >= 10:
                                    # Sort values for percentile calculation
                                    sorted_values = sorted(numeric_values)
                                    length = len(sorted_values)

                                    stats.update(
                                        {
                                            "p50": sorted_values[int(length * 0.5)],
                                            "p90": sorted_values[int(length * 0.9)],
                                            "p95": sorted_values[int(length * 0.95)],
                                            "p99": (
                                                sorted_values[int(length * 0.99)]
                                                if length >= 100
                                                else None
                                            ),
                                        }
                                    )

                            aggregated[provider][service][operation][metric_type] = stats

            self.aggregated = aggregated

    def get_metrics(
        self,
        provider: Union[CloudProvider, str, None] = None,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        metric_type: Union[MetricType, str, None] = None,
        aggregated: bool = True,
    ) -> Dict[str, Any]:
        """
        Get metrics for specified criteria.

        Args:
            provider: Cloud provider filter
            service: Service name filter
            operation: Operation name filter
            metric_type: Type of metric filter
            aggregated: Whether to return aggregated statistics

        Returns:
            Dictionary of metrics
        """
        # Force aggregation if needed
        if aggregated and time.time() - self.last_aggregation >= self.aggregation_interval:
            self._aggregate_metrics()

        # Normalize provider
        if provider is not None:
            if isinstance(provider, str):
                try:
                    provider_value = CloudProvider(provider.lower()).value
                except ValueError:
                    provider_value = provider.lower()
            else:
                provider_value = provider.value
        else:
            provider_value = None

        # Normalize metric type
        if metric_type is not None:
            if isinstance(metric_type, str):
                try:
                    metric_type_value = MetricType(metric_type.lower()).value
                except ValueError:
                    metric_type_value = metric_type.lower()
            else:
                metric_type_value = metric_type.value
        else:
            metric_type_value = None

        # Get data source
        data_source = self.aggregated if aggregated else self.metrics

        # Apply filters
        result = {}

        with self.lock:
            # Filter by provider
            if provider_value:
                if provider_value in data_source:
                    providers = {provider_value: data_source[provider_value]}
                else:
                    return {}
            else:
                providers = data_source

            # Copy filtered data
            for p in providers:
                result[p] = {}

                # Filter by service
                if service:
                    if service in providers[p]:
                        services = {service: providers[p][service]}
                    else:
                        continue
                else:
                    services = providers[p]

                for s in services:
                    result[p][s] = {}

                    # Filter by operation
                    if operation:
                        if operation in services[s]:
                            operations = {operation: services[s][operation]}
                        else:
                            continue
                    else:
                        operations = services[s]

                    for o in operations:
                        result[p][s][o] = {}

                        # Filter by metric type
                        if metric_type_value:
                            if metric_type_value in operations[o]:
                                result[p][s][o][metric_type_value] = operations[o][
                                    metric_type_value
                                ]
                        else:
                            result[p][s][o] = operations[o]

        return result

    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        with self.lock:
            # Re-initialize metrics storage
            self.metrics = {
                provider.value: {
                    service: {
                        operation: {metric_type.value: [] for metric_type in MetricType}
                        for operation in ["all"]
                    }
                    for service in ["all"]
                }
                for provider in CloudProvider
            }

            # Reset aggregated metrics
            self.aggregated = {}

            # Reset last aggregation time
            self.last_aggregation = time.time()

    def enable(self) -> None:
        """Enable metrics collection."""
        self.enabled = True

    def disable(self) -> None:
        """Disable metrics collection."""
        self.enabled = False

    def set_retention_period(self, seconds: int) -> None:
        """Set the metrics retention period in seconds."""
        self.retention_period = max(1, seconds)

    def set_aggregation_interval(self, seconds: int) -> None:
        """Set the metrics aggregation interval in seconds."""
        self.aggregation_interval = max(1, seconds)

    def set_max_samples(self, samples: int) -> None:
        """Set the maximum number of samples per metric."""
        self.max_samples = max(10, samples)

    def generate_report(
        self,
        provider: Union[CloudProvider, str, None] = None,
        service: Optional[str] = None,
        period: Optional[int] = None,
        format_type: str = "text",
    ) -> str:
        """
        Generate a performance report.

        Args:
            provider: Cloud provider filter
            service: Service name filter
            period: Report period in seconds (None for all available data)
            format_type: Output format ('text' or 'json')

        Returns:
            Formatted report as string
        """
        # Force aggregation
        self._aggregate_metrics()

        # Get metrics
        metrics = self.get_metrics(provider=provider, service=service, aggregated=True)

        if format_type.lower() == "json":
            return json.dumps(metrics, indent=2, default=str)
        else:
            # Generate text report
            report = []
            report.append("=" * 80)
            report.append("CLOUD SERVICE PERFORMANCE REPORT")
            report.append("=" * 80)

            timestamp = DateProvider.get_instance().now().strftime("%Y-%m-%d %H:%M:%S")
            report.append(f"Generated: {timestamp}")

            if provider:
                report.append(f"Provider: {provider}")
            if service:
                report.append(f"Service: {service}")

            report.append("-" * 80)

            # Add metrics sections
            for p in metrics:
                report.append(f"\nProvider: {p.upper()}")

                for s in metrics[p]:
                    report.append(f"\n  Service: {s}")

                    for o in metrics[p][s]:
                        report.append(f"\n    Operation: {o}")

                        for m in metrics[p][s][o]:
                            stats = metrics[p][s][o][m]
                            report.append(f"\n      {m.upper()}:")

                            for stat_name, stat_value in stats.items():
                                if stat_value is not None:
                                    if (
                                        stat_name in ["mean", "p50", "p90", "p95", "p99"]
                                        and m == MetricType.LATENCY.value
                                    ):
                                        # Format latency as milliseconds
                                        report.append(
                                            f"        {stat_name}: {stat_value * 1000:.2f} ms"
                                        )
                                    elif (
                                        stat_name in ["mean", "p50", "p90", "p95", "p99"]
                                        and m == MetricType.ERROR_RATE.value
                                    ):
                                        # Format error rate as percentage
                                        report.append(
                                            f"        {stat_name}: {stat_value * 100:.2f}%"
                                        )
                                    else:
                                        report.append(f"        {stat_name}: {stat_value}")

            return "\n".join(report)


# Global metrics manager instance
_metrics_manager = MetricsManager()


def get_metrics_manager() -> MetricsManager:
    """Get the global metrics manager instance."""
    return _metrics_manager


def record_metric(
    provider: Union[CloudProvider, str],
    service: str,
    operation: str,
    metric_type: Union[MetricType, str],
    value: Any,
) -> None:
    """
    Record a metric value (convenience function).

    Args:
        provider: Cloud provider
        service: Service name
        operation: Operation name
        metric_type: Type of metric
        value: Metric value
    """
    _metrics_manager.record_metric(provider, service, operation, metric_type, value)


def track_performance(
    provider: Union[CloudProvider, str], service: str, operation: Optional[str] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to track performance metrics for cloud operations.

    Args:
        provider: Cloud provider
        service: Service name
        operation: Operation name (defaults to function name)

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            op_name = operation or func.__name__
            start_time = time.time()
            success = True
            retry_count = 0

            # Check for retry_count in kwargs
            if "_retry_attempt" in kwargs:
                retry_count = kwargs.pop("_retry_attempt")

            try:
                result = func(*args, **kwargs)

                if isinstance(result, dict) and "success" in result:
                    success = result["success"]

                return result

            except Exception:
                success = False
                _metrics_manager.record_metric(
                    provider, service, op_name, MetricType.ERROR_RATE, 1.0
                )
                raise

            finally:
                # Record latency
                latency = time.time() - start_time
                _metrics_manager.record_metric(
                    provider, service, op_name, MetricType.LATENCY, latency
                )

                # Record success rate
                _metrics_manager.record_metric(
                    provider,
                    service,
                    op_name,
                    MetricType.SUCCESS_RATE,
                    1.0 if success else 0.0,
                )

                # Record retry count if available
                if retry_count > 0:
                    _metrics_manager.record_metric(
                        provider, service, op_name, MetricType.RETRY_COUNT, retry_count
                    )

        return wrapper

    return decorator
