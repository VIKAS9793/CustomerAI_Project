"""
Chaos testing utilities for cloud services.

This module provides tools for testing the resilience of cloud services
by introducing controlled failures and degradations.

Copyright (c) 2025 Vikas Sahani
GitHub: https://github.com/VIKAS9793
Email: vikassahani17@gmail.com

Licensed under MIT License - see LICENSE file for details
This copyright and license applies only to the original code in this file,
not to any third-party libraries or dependencies used.
"""

import functools
import json
import logging
import random
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from cloud.config import CloudProvider
from cloud.errors import CloudNetworkError, CloudServiceUnavailableError, CloudTimeoutError

logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar("T")


class FailureMode(Enum):
    """Types of failures that can be injected."""

    LATENCY = "latency"  # Increased latency
    ERROR = "error"  # Return error
    THROTTLE = "throttle"  # Rate limiting
    TIMEOUT = "timeout"  # Operation timeout
    DROP = "drop"  # Drop request (no response)


class InjectionScope(Enum):
    """Scope of failure injection."""

    ALL = "all"  # All operations
    PROVIDER = "provider"  # Specific provider
    SERVICE = "service"  # Specific service
    OPERATION = "operation"  # Specific operation


class ChaosTestingEngine:
    """
    Engine for chaos testing cloud services.

    This class provides methods for injecting failures into
    cloud service operations in a controlled manner.
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
                    cls._instance = super(ChaosTestingEngine, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize chaos testing engine (only once for singleton)."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Active failure injections
        self.injections: Dict[str, Dict[str, Any]] = {}

        # Active experiments
        self.experiments: Dict[str, Dict[str, Any]] = {}

        # Configuration
        self.enabled = False  # Disabled by default for safety
        self.dry_run = False  # If True, log but don't actually inject failures

        # Thread safety
        import threading

        self.lock = threading.RLock()

        self._initialized = True

    def enable(self) -> None:
        """Enable chaos testing engine."""
        with self.lock:
            self.enabled = True
            logger.info("Chaos testing engine enabled")

    def disable(self) -> None:
        """Disable chaos testing engine."""
        with self.lock:
            self.enabled = False
            logger.info("Chaos testing engine disabled")

    def set_dry_run(self, dry_run: bool) -> None:
        """
        Set dry run mode.

        In dry run mode, failures are logged but not actually injected.

        Args:
            dry_run: Whether to enable dry run mode
        """
        with self.lock:
            self.dry_run = dry_run
            logger.info(f"Chaos testing engine dry run mode: {dry_run}")

    def add_injection(
        self,
        injection_id: str,
        failure_mode: FailureMode,
        scope: InjectionScope,
        target: Optional[str] = None,
        target_secondary: Optional[str] = None,
        probability: float = 1.0,
        duration: Optional[float] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a failure injection.

        Args:
            injection_id: Unique identifier for the injection
            failure_mode: Type of failure to inject
            scope: Scope of the injection
            target: Primary target (provider, service, or operation)
            target_secondary: Secondary target (service or operation)
            probability: Probability of failure (0.0-1.0)
            duration: Duration of injection in seconds, or None for indefinite
            parameters: Additional parameters for the failure
        """
        with self.lock:
            # Create injection
            injection = {
                "id": injection_id,
                "failure_mode": failure_mode,
                "scope": scope,
                "target": target,
                "target_secondary": target_secondary,
                "probability": max(0.0, min(1.0, probability)),  # Clamp to [0.0, 1.0]
                "parameters": parameters or {},
                "created_at": time.time(),
                "expires_at": time.time() + duration if duration is not None else None,
                "hit_count": 0,
                "trigger_count": 0,
            }

            # Add injection
            self.injections[injection_id] = injection

            logger.info(f"Added failure injection: {injection_id} ({failure_mode.value})")

    def remove_injection(self, injection_id: str) -> None:
        """
        Remove a failure injection.

        Args:
            injection_id: Identifier of the injection to remove
        """
        with self.lock:
            if injection_id in self.injections:
                del self.injections[injection_id]
                logger.info(f"Removed failure injection: {injection_id}")

    def clear_injections(self) -> None:
        """Remove all failure injections."""
        with self.lock:
            self.injections.clear()
            logger.info("Cleared all failure injections")

    def get_injections(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active injections.

        Returns:
            Dictionary of injections
        """
        with self.lock:
            # Make a copy to avoid concurrent modification
            return {k: v.copy() for k, v in self.injections.items()}

    def get_injection(self, injection_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific injection.

        Args:
            injection_id: Identifier of the injection

        Returns:
            Injection data, or None if not found
        """
        with self.lock:
            return self.injections.get(injection_id, {}).copy()

    def _should_inject_failure(
        self, provider: str, service: str, operation: str
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if failure should be injected for a specific operation.

        Args:
            provider: Cloud provider
            service: Service name
            operation: Operation name

        Returns:
            Tuple of (should_inject, injection_data)
        """
        if not self.enabled:
            return False, None

        with self.lock:
            # Check for expired injections and remove them
            current_time = time.time()
            for injection_id in list(self.injections.keys()):
                injection = self.injections[injection_id]
                if injection["expires_at"] is not None and current_time > injection["expires_at"]:
                    del self.injections[injection_id]
                    logger.info(f"Removed expired injection: {injection_id}")

            # Check all active injections
            for injection_id, injection in self.injections.items():
                # Increment hit count
                injection["hit_count"] += 1

                # Check scope
                scope = injection["scope"]
                if scope == InjectionScope.ALL:
                    # All operations
                    match = True
                elif scope == InjectionScope.PROVIDER:
                    # Match provider
                    match = injection["target"].lower() == provider.lower()
                elif scope == InjectionScope.SERVICE:
                    # Match provider and service
                    match = (
                        injection["target"].lower() == provider.lower()
                        and injection["target_secondary"].lower() == service.lower()
                    )
                elif scope == InjectionScope.OPERATION:
                    # Match provider, service, and operation
                    match = (
                        injection["target"].lower() == provider.lower()
                        and injection["target_secondary"].lower() == service.lower()
                        and injection["parameters"].get("operation", "").lower()
                        == operation.lower()
                    )
                else:
                    match = False

                if match:
                    # Check probability
                    if random.random() < injection["probability"]:
                        # Increment trigger count
                        injection["trigger_count"] += 1
                        return True, injection

            return False, None

    def _inject_failure(
        self, provider: str, service: str, operation: str, injection: Dict[str, Any]
    ) -> None:
        """
        Inject a failure based on the injection configuration.

        Args:
            provider: Cloud provider
            service: Service name
            operation: Operation name
            injection: Injection configuration

        Raises:
            Exception: Various exceptions based on the failure mode
        """
        failure_mode = injection["failure_mode"]
        parameters = injection["parameters"]

        # Log the injection
        logger.info(f"Injecting failure: {provider}/{service}/{operation} - {failure_mode.value}")

        if self.dry_run:
            logger.info("DRY RUN: Failure not actually injected")
            return

        if failure_mode == FailureMode.LATENCY:
            # Inject latency
            latency = parameters.get("latency", 1.0)
            jitter = parameters.get("jitter", 0.1)

            # Add jitter to latency
            actual_latency = latency * (1.0 + random.uniform(-jitter, jitter))

            # Sleep to simulate latency
            time.sleep(actual_latency)

        elif failure_mode == FailureMode.ERROR:
            # Inject error
            error_type = parameters.get("error_type", "service_unavailable")
            error_message = parameters.get(
                "error_message",
                f"Chaos testing injected failure for {provider}/{service}/{operation}",
            )

            if error_type == "service_unavailable":
                raise CloudServiceUnavailableError(error_message)
            elif error_type == "network":
                raise CloudNetworkError(error_message)
            else:
                # Generic error
                raise Exception(error_message)

        elif failure_mode == FailureMode.THROTTLE:
            # Inject throttling
            if random.random() < parameters.get("throttle_probability", 0.5):
                error_message = parameters.get(
                    "error_message",
                    f"Chaos testing injected throttling for {provider}/{service}/{operation}",
                )
                raise CloudServiceUnavailableError(error_message)

        elif failure_mode == FailureMode.TIMEOUT:
            # Inject timeout
            timeout = parameters.get("timeout", 30.0)

            # Sleep to simulate timeout
            time.sleep(timeout)

            # Raise timeout error
            error_message = parameters.get(
                "error_message",
                f"Chaos testing injected timeout for {provider}/{service}/{operation}",
            )
            raise CloudTimeoutError(error_message)

        elif failure_mode == FailureMode.DROP:
            # Inject dropped request (hang indefinitely)
            # In practice, this will eventually be interrupted by a client timeout
            timeout = parameters.get("timeout", 3600.0)  # Default 1 hour
            time.sleep(timeout)

    def start_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str,
        injections: List[Dict[str, Any]],
        duration: float,
        stagger_start: Optional[float] = None,
    ) -> None:
        """
        Start a chaos experiment with multiple injections.

        Args:
            experiment_id: Unique identifier for the experiment
            name: Experiment name
            description: Experiment description
            injections: List of injection configurations
            duration: Total duration of the experiment in seconds
            stagger_start: Optional delay between starting injections
        """
        with self.lock:
            # Create experiment
            experiment = {
                "id": experiment_id,
                "name": name,
                "description": description,
                "injections": [],
                "created_at": time.time(),
                "expires_at": time.time() + duration,
                "status": "running",
            }

            # Add experiment
            self.experiments[experiment_id] = experiment

            # Enable chaos testing
            self.enabled = True

            logger.info(f"Started chaos experiment: {experiment_id} - {name}")

            # Add injections
            for i, injection_config in enumerate(injections):
                # Generate injection ID if not provided
                injection_id = injection_config.get("id", f"{experiment_id}_injection_{i}")

                # Set expiry to match experiment if not provided
                if "duration" not in injection_config:
                    injection_config["duration"] = duration

                # Extract injection parameters
                failure_mode = FailureMode(injection_config.get("failure_mode", "error"))
                scope = InjectionScope(injection_config.get("scope", "all"))
                target = injection_config.get("target")
                target_secondary = injection_config.get("target_secondary")
                probability = injection_config.get("probability", 1.0)
                inj_duration = injection_config.get("duration")
                parameters = injection_config.get("parameters", {})

                # Calculate staggered start time if needed
                if stagger_start and i > 0:
                    start_delay = i * stagger_start

                    # Create a delayed injection using a background thread
                    def delayed_injection(delay):
                        time.sleep(delay)
                        self.add_injection(
                            injection_id=injection_id,
                            failure_mode=failure_mode,
                            scope=scope,
                            target=target,
                            target_secondary=target_secondary,
                            probability=probability,
                            duration=inj_duration,
                            parameters=parameters,
                        )
                        logger.info(f"Added delayed injection: {injection_id} after {delay:.2f}s")

                    # Start delayed injection thread
                    threading.Thread(
                        target=delayed_injection, args=(start_delay,), daemon=True
                    ).start()

                    # Record injection in experiment
                    experiment["injections"].append(
                        {"id": injection_id, "delayed": True, "delay": start_delay}
                    )

                else:
                    # Add injection immediately
                    self.add_injection(
                        injection_id=injection_id,
                        failure_mode=failure_mode,
                        scope=scope,
                        target=target,
                        target_secondary=target_secondary,
                        probability=probability,
                        duration=inj_duration,
                        parameters=parameters,
                    )

                    # Record injection in experiment
                    experiment["injections"].append({"id": injection_id, "delayed": False})

            # Schedule experiment cleanup
            def cleanup_experiment():
                time.sleep(duration)
                with self.lock:
                    if experiment_id in self.experiments:
                        experiment = self.experiments[experiment_id]
                        experiment["status"] = "completed"

                        # Check if all injections were added
                        all_injections_added = True
                        for inj in experiment["injections"]:
                            if (
                                inj["delayed"]
                                and time.time() < experiment["created_at"] + inj["delay"]
                            ):
                                all_injections_added = False
                                break

                        if all_injections_added:
                            logger.info(f"Completed chaos experiment: {experiment_id}")
                        else:
                            logger.warning(
                                f"Chaos experiment {experiment_id} completed before all injections were added"
                            )

            # Start cleanup thread
            threading.Thread(target=cleanup_experiment, daemon=True).start()

    def stop_experiment(self, experiment_id: str) -> None:
        """
        Stop a running experiment.

        Args:
            experiment_id: Identifier of the experiment to stop
        """
        with self.lock:
            if experiment_id in self.experiments:
                experiment = self.experiments[experiment_id]

                # Remove all injections associated with this experiment
                for injection in experiment["injections"]:
                    injection_id = injection["id"]
                    self.remove_injection(injection_id)

                # Update experiment status
                experiment["status"] = "stopped"

                logger.info(f"Stopped chaos experiment: {experiment_id}")

    def get_experiments(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all experiments.

        Returns:
            Dictionary of experiments
        """
        with self.lock:
            # Make a copy to avoid concurrent modification
            return {k: v.copy() for k, v in self.experiments.items()}

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific experiment.

        Args:
            experiment_id: Identifier of the experiment

        Returns:
            Experiment data, or None if not found
        """
        with self.lock:
            return self.experiments.get(experiment_id, {}).copy()

    def get_experiment_report(self, experiment_id: str) -> Dict[str, Any]:
        """
        Generate a report for a specific experiment.

        Args:
            experiment_id: Identifier of the experiment

        Returns:
            Experiment report
        """
        with self.lock:
            if experiment_id not in self.experiments:
                return {"error": f"Experiment not found: {experiment_id}"}

            experiment = self.experiments[experiment_id]

            # Gather injection data
            injection_data = []
            for injection_info in experiment["injections"]:
                injection_id = injection_info["id"]
                injection = self.injections.get(injection_id, {})

                if injection:
                    # Injection still active
                    injection_data.append(
                        {
                            "id": injection_id,
                            "failure_mode": (
                                str(injection["failure_mode"].value)
                                if "failure_mode" in injection
                                else "unknown"
                            ),
                            "hit_count": injection.get("hit_count", 0),
                            "trigger_count": injection.get("trigger_count", 0),
                            "trigger_rate": injection.get("trigger_count", 0)
                            / max(1, injection.get("hit_count", 0)),
                        }
                    )
                else:
                    # Injection completed or never started
                    injection_data.append(
                        {
                            "id": injection_id,
                            "status": (
                                "completed" if injection_info.get("delayed", False) else "missing"
                            ),
                        }
                    )

            # Create report
            report = {
                "id": experiment_id,
                "name": experiment.get("name", ""),
                "description": experiment.get("description", ""),
                "status": experiment.get("status", "unknown"),
                "created_at": experiment.get("created_at", 0),
                "expires_at": experiment.get("expires_at", 0),
                "duration": experiment.get("expires_at", 0) - experiment.get("created_at", 0),
                "elapsed": time.time() - experiment.get("created_at", time.time()),
                "remaining": max(0, experiment.get("expires_at", 0) - time.time()),
                "injections": injection_data,
            }

            return report

    def save_experiment_report(self, experiment_id: str, filename: str) -> None:
        """
        Save an experiment report to a file.

        Args:
            experiment_id: Identifier of the experiment
            filename: File to save report to
        """
        report = self.get_experiment_report(experiment_id)

        try:
            with open(filename, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Saved experiment report to {filename}")

        except Exception as e:
            logger.error(f"Error saving experiment report: {str(e)}")


# Global chaos testing engine instance
_chaos_engine = ChaosTestingEngine()


def get_chaos_engine() -> ChaosTestingEngine:
    """Get the global chaos testing engine instance."""
    return _chaos_engine


def with_chaos_testing(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to enable chaos testing for a function.

    This decorator checks if a failure should be injected
    for the function and injects it if needed.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        # Get chaos engine
        engine = get_chaos_engine()

        # Skip if chaos testing is disabled
        if not engine.enabled:
            return func(*args, **kwargs)

        # Extract provider, service, and operation information
        provider = kwargs.get("provider")
        service = kwargs.get("service")
        operation = kwargs.get("operation")

        # If provider/service/operation not provided in kwargs, try to extract from args
        if provider is None and len(args) > 0:
            provider = args[0]

        if service is None and len(args) > 1:
            service = args[1]

        if operation is None and len(args) > 2:
            operation = args[2]

        # Default values for provider/service/operation
        provider = str(provider) if provider is not None else "unknown"
        service = str(service) if service is not None else "unknown"
        operation = str(operation) if operation is not None else func.__name__

        # Check if we should inject a failure
        should_inject, injection = engine._should_inject_failure(provider, service, operation)

        if should_inject and injection:
            # Inject failure
            engine._inject_failure(provider, service, operation, injection)

        # If no injection or dry run, call the original function
        return func(*args, **kwargs)

    return wrapper


def create_chaos_experiment(
    name: str,
    description: str,
    injections: List[Dict[str, Any]],
    duration: float,
    stagger_start: Optional[float] = None,
) -> str:
    """
    Create and start a new chaos experiment.

    Args:
        name: Experiment name
        description: Experiment description
        injections: List of injection configurations
        duration: Total duration of the experiment in seconds
        stagger_start: Optional delay between starting injections

    Returns:
        Experiment ID
    """
    import uuid

    # Generate experiment ID
    experiment_id = str(uuid.uuid4())

    # Start experiment
    get_chaos_engine().start_experiment(
        experiment_id=experiment_id,
        name=name,
        description=description,
        injections=injections,
        duration=duration,
        stagger_start=stagger_start,
    )

    return experiment_id


def inject_failure(
    failure_mode: Union[FailureMode, str],
    scope: Union[InjectionScope, str] = InjectionScope.ALL,
    target: Optional[str] = None,
    target_secondary: Optional[str] = None,
    probability: float = 1.0,
    duration: Optional[float] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Inject a failure.

    Args:
        failure_mode: Type of failure to inject
        scope: Scope of the injection
        target: Primary target (provider, service, or operation)
        target_secondary: Secondary target (service or operation)
        probability: Probability of failure (0.0-1.0)
        duration: Duration of injection in seconds, or None for indefinite
        parameters: Additional parameters for the failure

    Returns:
        Injection ID
    """
    import uuid

    # Generate injection ID
    injection_id = str(uuid.uuid4())

    # Convert string enums to enum values if needed
    if isinstance(failure_mode, str):
        failure_mode = FailureMode(failure_mode)

    if isinstance(scope, str):
        scope = InjectionScope(scope)

    # Add injection
    get_chaos_engine().add_injection(
        injection_id=injection_id,
        failure_mode=failure_mode,
        scope=scope,
        target=target,
        target_secondary=target_secondary,
        probability=probability,
        duration=duration,
        parameters=parameters,
    )

    return injection_id


def run_standard_chaos_test(
    provider: Union[CloudProvider, str],
    service: str,
    duration: float = 60.0,
    stagger: bool = True,
) -> str:
    """
    Run a standard chaos test for a specific service.

    This creates an experiment with standard failure injections
    for a service:
    - Latency
    - Errors
    - Throttling

    Args:
        provider: Cloud provider
        service: Service name
        duration: Test duration in seconds
        stagger: Whether to stagger injection start times

    Returns:
        Experiment ID
    """
    # Normalize provider
    if isinstance(provider, CloudProvider):
        provider_str = provider.value
    else:
        provider_str = str(provider).lower()

    # Create injections
    injections = [
        # Latency injection (50% probability, 1-3 seconds)
        {
            "failure_mode": FailureMode.LATENCY.value,
            "scope": InjectionScope.SERVICE.value,
            "target": provider_str,
            "target_secondary": service,
            "probability": 0.5,
            "parameters": {"latency": 2.0, "jitter": 0.5},
        },
        # Error injection (10% probability)
        {
            "failure_mode": FailureMode.ERROR.value,
            "scope": InjectionScope.SERVICE.value,
            "target": provider_str,
            "target_secondary": service,
            "probability": 0.1,
            "parameters": {"error_type": "service_unavailable"},
        },
        # Throttling injection (20% probability)
        {
            "failure_mode": FailureMode.THROTTLE.value,
            "scope": InjectionScope.SERVICE.value,
            "target": provider_str,
            "target_secondary": service,
            "probability": 0.2,
            "parameters": {"throttle_probability": 0.5},
        },
    ]

    # Create experiment
    stagger_start = 10.0 if stagger else None

    return create_chaos_experiment(
        name=f"Standard chaos test for {provider_str}/{service}",
        description=f"Standard chaos test with latency, errors, and throttling for {provider_str}/{service}",
        injections=injections,
        duration=duration,
        stagger_start=stagger_start,
    )
