"""
Prometheus integration for cloud services observability.

This module provides integration with Prometheus for monitoring
cloud service operations.

Copyright (c) 2025 Vikas Sahani
GitHub: https://github.com/VIKAS9793
Email: vikassahani17@gmail.com

Licensed under MIT License - see LICENSE file for details
This copyright and license applies only to the original code in this file,
not to any third-party libraries or dependencies used.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar
from enum import Enum
import functools
import socket
import http.server
import socketserver
from urllib.parse import parse_qs, urlparse

from cloud.config import CloudProvider
from cloud.utils.metrics import get_metrics_manager, MetricType

# Try to import prometheus_client library, providing helpful error if not installed
try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary
    from prometheus_client.core import CollectorRegistry, REGISTRY
    PROMETHEUS_CLIENT_AVAILABLE = True
except ImportError:
    PROMETHEUS_CLIENT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar('T')

class MetricCategory(Enum):
    """Categories of metrics for Prometheus."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class PrometheusExporter:
    """
    Exporter for Prometheus metrics.
    
    This class converts internal metrics to Prometheus metrics
    and exposes them via HTTP.
    """
    
    _instance = None
    _lock = None
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            import threading
            cls._lock = threading.Lock()
            with cls._lock:
                if cls._instance is None:
                    if not PROMETHEUS_CLIENT_AVAILABLE:
                        raise ImportError(
                            "Prometheus integration requires the 'prometheus_client' package. "
                            "Please install it with 'pip install prometheus_client'."
                        )
                    cls._instance = super(PrometheusExporter, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize Prometheus exporter (only once for singleton)."""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # Registry for metrics
        self.registry = CollectorRegistry()
        
        # Metrics storage
        self.metrics: Dict[str, Dict[str, Any]] = {}
        
        # HTTP server
        self.server = None
        self.server_thread = None
        self.server_port = 9090
        
        # Configuration
        self.enabled = True
        self.export_interval = 15  # seconds
        self.metric_prefix = "cloud"
        
        # Thread safety
        import threading
        self.lock = threading.RLock()
        
        # Export thread
        self._export_thread_running = False
        
        self._initialized = True
        
        # Register default metrics
        self._register_default_metrics()
    
    def _register_default_metrics(self):
        """Register default metrics."""
        # Process metrics
        if hasattr(prometheus_client, 'process_collector'):
            prometheus_client.process_collector.ProcessCollector(registry=self.registry)
        
        # Platform metrics
        if hasattr(prometheus_client, 'platform_collector'):
            prometheus_client.platform_collector.PlatformCollector(registry=self.registry)
    
    def _get_metric_name(self, provider: str, service: str, operation: str, metric_type: str) -> str:
        """
        Generate a Prometheus metric name.
        
        Args:
            provider: Cloud provider
            service: Service name
            operation: Operation name
            metric_type: Metric type
            
        Returns:
            Prometheus metric name
        """
        return f"{self.metric_prefix}_{provider}_{service}_{operation}_{metric_type}"
    
    def _get_or_create_metric(
        self,
        provider: str,
        service: str,
        operation: str,
        metric_type: str,
        category: MetricCategory,
        description: str = "",
        labels: Optional[List[str]] = None
    ) -> Any:
        """
        Get or create a Prometheus metric.
        
        Args:
            provider: Cloud provider
            service: Service name
            operation: Operation name
            metric_type: Metric type
            category: Metric category
            description: Metric description
            labels: Metric labels
            
        Returns:
            Prometheus metric
        """
        metric_name = self._get_metric_name(provider, service, operation, metric_type)
        
        with self.lock:
            if metric_name in self.metrics:
                return self.metrics[metric_name]["metric"]
            
            # Create metric based on category
            if category == MetricCategory.COUNTER:
                metric = Counter(
                    metric_name,
                    description,
                    labels or [],
                    registry=self.registry
                )
            elif category == MetricCategory.GAUGE:
                metric = Gauge(
                    metric_name,
                    description,
                    labels or [],
                    registry=self.registry
                )
            elif category == MetricCategory.HISTOGRAM:
                metric = Histogram(
                    metric_name,
                    description,
                    labels or [],
                    registry=self.registry
                )
            elif category == MetricCategory.SUMMARY:
                metric = Summary(
                    metric_name,
                    description,
                    labels or [],
                    registry=self.registry
                )
            else:
                raise ValueError(f"Unsupported metric category: {category}")
            
            # Store metric
            self.metrics[metric_name] = {
                "metric": metric,
                "category": category,
                "labels": labels or []
            }
            
            return metric
    
    def export_metrics(self) -> None:
        """
        Export metrics from internal metrics manager to Prometheus.
        """
        if not self.enabled or not PROMETHEUS_CLIENT_AVAILABLE:
            return
        
        try:
            # Get metrics manager
            metrics_manager = get_metrics_manager()
            
            # Get all metrics
            metrics_data = metrics_manager.get_metrics(aggregated=True)
            
            # Convert to Prometheus metrics
            for provider in metrics_data:
                for service in metrics_data[provider]:
                    for operation in metrics_data[provider][service]:
                        for metric_type_str in metrics_data[provider][service][operation]:
                            metric_data = metrics_data[provider][service][operation][metric_type_str]
                            
                            # Skip if no data
                            if not metric_data:
                                continue
                            
                            # Determine appropriate Prometheus metric type
                            if metric_type_str == MetricType.LATENCY.value:
                                # Latency as histogram
                                metric = self._get_or_create_metric(
                                    provider=provider,
                                    service=service,
                                    operation=operation,
                                    metric_type="latency_seconds",
                                    category=MetricCategory.HISTOGRAM,
                                    description=f"Latency of {provider} {service} {operation} operations",
                                )
                                
                                # Update histogram with latest value
                                if "last_value" in metric_data:
                                    metric.observe(metric_data["last_value"])
                                
                            elif metric_type_str == MetricType.ERROR_RATE.value:
                                # Error rate as gauge
                                metric = self._get_or_create_metric(
                                    provider=provider,
                                    service=service,
                                    operation=operation,
                                    metric_type="error_rate",
                                    category=MetricCategory.GAUGE,
                                    description=f"Error rate of {provider} {service} {operation} operations",
                                )
                                
                                # Update gauge with latest value
                                if "last_value" in metric_data:
                                    metric.set(metric_data["last_value"])
                                
                            elif metric_type_str == MetricType.THROUGHPUT.value:
                                # Throughput as gauge
                                metric = self._get_or_create_metric(
                                    provider=provider,
                                    service=service,
                                    operation=operation,
                                    metric_type="throughput",
                                    category=MetricCategory.GAUGE,
                                    description=f"Throughput of {provider} {service} {operation} operations",
                                )
                                
                                # Update gauge with latest value
                                if "last_value" in metric_data:
                                    metric.set(metric_data["last_value"])
                                
                            elif metric_type_str == MetricType.SUCCESS_RATE.value:
                                # Success rate as gauge
                                metric = self._get_or_create_metric(
                                    provider=provider,
                                    service=service,
                                    operation=operation,
                                    metric_type="success_rate",
                                    category=MetricCategory.GAUGE,
                                    description=f"Success rate of {provider} {service} {operation} operations",
                                )
                                
                                # Update gauge with latest value
                                if "last_value" in metric_data:
                                    metric.set(metric_data["last_value"])
                                
                            elif metric_type_str == MetricType.RETRY_COUNT.value:
                                # Retry count as counter
                                metric = self._get_or_create_metric(
                                    provider=provider,
                                    service=service,
                                    operation=operation,
                                    metric_type="retry_total",
                                    category=MetricCategory.COUNTER,
                                    description=f"Total retries of {provider} {service} {operation} operations",
                                )
                                
                                # Update counter with latest value
                                if "last_value" in metric_data and "count" in metric_data:
                                    # Increment counter by difference between current count and previous count
                                    metric._value.inc(metric_data["last_value"])
                                
                            else:
                                # Default to gauge for unknown metric types
                                metric = self._get_or_create_metric(
                                    provider=provider,
                                    service=service,
                                    operation=operation,
                                    metric_type=metric_type_str,
                                    category=MetricCategory.GAUGE,
                                    description=f"{metric_type_str} of {provider} {service} {operation} operations",
                                )
                                
                                # Update gauge with latest value
                                if "last_value" in metric_data:
                                    metric.set(metric_data["last_value"])
            
            logger.debug("Exported metrics to Prometheus")
            
        except Exception as e:
            logger.error(f"Error exporting metrics to Prometheus: {str(e)}")
    
    def start_export_thread(self) -> None:
        """Start the metrics export background thread."""
        if not self._export_thread_running and self.enabled and PROMETHEUS_CLIENT_AVAILABLE:
            import threading
            export_thread = threading.Thread(
                target=self._export_task,
                daemon=True
            )
            export_thread.start()
            self._export_thread_running = True
            
            logger.info(f"Started Prometheus metrics export thread (interval: {self.export_interval}s)")
    
    def _export_task(self) -> None:
        """Background task for exporting metrics."""
        while self.enabled:
            try:
                self.export_metrics()
            except Exception as e:
                logger.error(f"Error in Prometheus metrics export: {str(e)}")
            
            # Sleep for export interval
            time.sleep(self.export_interval)
    
    def start_http_server(self, port: int = 9090) -> None:
        """
        Start HTTP server for Prometheus metrics.
        
        Args:
            port: HTTP server port
        """
        if not PROMETHEUS_CLIENT_AVAILABLE:
            logger.error("Cannot start Prometheus HTTP server: prometheus_client not available")
            return
        
        with self.lock:
            if self.server:
                logger.warning("Prometheus HTTP server already running")
                return
            
            self.server_port = port
            
            try:
                # Start HTTP server in a separate thread
                import threading
                
                class PrometheusHandler(http.server.BaseHTTPRequestHandler):
                    def do_GET(self):
                        url = urlparse(self.path)
                        if url.path == '/metrics':
                            self.send_response(200)
                            self.send_header('Content-Type', 'text/plain')
                            self.end_headers()
                            output = prometheus_client.generate_latest(self.server.registry)
                            self.wfile.write(output)
                        else:
                            self.send_response(404)
                            self.end_headers()
                            self.wfile.write(b'Not Found')
                    
                    def log_message(self, format, *args):
                        # Suppress HTTP server logs
                        pass
                
                class PrometheusServer(socketserver.ThreadingTCPServer):
                    def __init__(self, server_address, RequestHandlerClass, registry):
                        self.registry = registry
                        socketserver.ThreadingTCPServer.__init__(self, server_address, RequestHandlerClass)
                
                # Create server
                self.server = PrometheusServer(('', port), PrometheusHandler, self.registry)
                
                # Start server in a separate thread
                self.server_thread = threading.Thread(
                    target=self.server.serve_forever,
                    daemon=True
                )
                self.server_thread.start()
                
                logger.info(f"Started Prometheus HTTP server on port {port}")
                
                # Start export thread if not already running
                self.start_export_thread()
                
            except Exception as e:
                logger.error(f"Error starting Prometheus HTTP server: {str(e)}")
                self.server = None
                self.server_thread = None
    
    def stop_http_server(self) -> None:
        """Stop HTTP server for Prometheus metrics."""
        with self.lock:
            if self.server:
                try:
                    self.server.shutdown()
                    self.server = None
                    self.server_thread = None
                    logger.info("Stopped Prometheus HTTP server")
                except Exception as e:
                    logger.error(f"Error stopping Prometheus HTTP server: {str(e)}")
    
    def enable(self) -> None:
        """Enable Prometheus metrics export."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable Prometheus metrics export."""
        self.enabled = False
        
        # Stop HTTP server if running
        self.stop_http_server()
    
    def set_export_interval(self, seconds: int) -> None:
        """
        Set the metrics export interval in seconds.
        
        Args:
            seconds: Export interval in seconds
        """
        self.export_interval = max(1, seconds)
    
    def set_metric_prefix(self, prefix: str) -> None:
        """
        Set the metric name prefix.
        
        Args:
            prefix: Metric name prefix
        """
        self.metric_prefix = prefix


# Global Prometheus exporter instance
_prometheus_exporter = PrometheusExporter() if PROMETHEUS_CLIENT_AVAILABLE else None

def get_prometheus_exporter() -> Optional[PrometheusExporter]:
    """Get the global Prometheus exporter instance."""
    return _prometheus_exporter


def start_prometheus_server(port: int = 9090) -> None:
    """
    Start Prometheus HTTP server on the specified port.
    
    This is a convenience function that starts both the HTTP server
    and the metrics export thread.
    
    Args:
        port: HTTP server port
    """
    if _prometheus_exporter is None:
        logger.error("Cannot start Prometheus server: prometheus_client not available")
        return
    
    _prometheus_exporter.start_http_server(port)


def stop_prometheus_server() -> None:
    """Stop Prometheus HTTP server."""
    if _prometheus_exporter is not None:
        _prometheus_exporter.stop_http_server()


def configure_prometheus_export(
    enabled: bool = True,
    export_interval: int = 15,
    metric_prefix: str = "cloud"
) -> None:
    """
    Configure Prometheus metrics export.
    
    Args:
        enabled: Whether to enable metrics export
        export_interval: Export interval in seconds
        metric_prefix: Metric name prefix
    """
    if _prometheus_exporter is None:
        logger.error("Cannot configure Prometheus export: prometheus_client not available")
        return
    
    _prometheus_exporter.enabled = enabled
    _prometheus_exporter.set_export_interval(export_interval)
    _prometheus_exporter.set_metric_prefix(metric_prefix)
    
    if enabled:
        _prometheus_exporter.start_export_thread()


def with_prometheus_metrics(
    name: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to track function execution with Prometheus metrics.
    
    This decorator creates Prometheus metrics for the function:
    - A histogram for execution time
    - A counter for total calls
    - A counter for failures
    
    Args:
        name: Optional metric name override
        labels: Optional metric labels
        
    Returns:
        Decorator function
    """
    if not PROMETHEUS_CLIENT_AVAILABLE:
        # Return no-op decorator if Prometheus is not available
        def no_op_decorator(func: Callable[..., T]) -> Callable[..., T]:
            return func
        
        return no_op_decorator
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Generate metric name from function name if not provided
        metric_name = name or func.__name__
        
        # Convert labels to list of label names
        label_names = list(labels.keys()) if labels else []
        
        # Create metrics
        duration_metric = Histogram(
            f"{metric_name}_duration_seconds",
            f"Duration of {metric_name} in seconds",
            label_names,
            registry=_prometheus_exporter.registry if _prometheus_exporter else REGISTRY
        )
        
        calls_metric = Counter(
            f"{metric_name}_calls_total",
            f"Total calls to {metric_name}",
            label_names,
            registry=_prometheus_exporter.registry if _prometheus_exporter else REGISTRY
        )
        
        failures_metric = Counter(
            f"{metric_name}_failures_total",
            f"Total failures of {metric_name}",
            label_names,
            registry=_prometheus_exporter.registry if _prometheus_exporter else REGISTRY
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Extract label values
            label_values = list(labels.values()) if labels else []
            
            # Increment calls counter
            calls_metric.labels(*label_values).inc()
            
            # Record start time
            start_time = time.time()
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                # Increment failures counter
                failures_metric.labels(*label_values).inc()
                raise
                
            finally:
                # Record duration
                duration = time.time() - start_time
                duration_metric.labels(*label_values).observe(duration)
        
        return wrapper
    
    return decorator 