"""
Load testing utilities for cloud services.

This module provides tools for testing the scalability and performance
of cloud services under load.

Copyright (c) 2025 Vikas Sahani
GitHub: https://github.com/VIKAS9793
Email: vikassahani17@gmail.com

Licensed under MIT License - see LICENSE file for details
This copyright and license applies only to the original code in this file,
not to any third-party libraries or dependencies used.
"""

import concurrent.futures
import csv
import json
import logging
import random
import statistics
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar("T")


class LoadPattern(Enum):
    """Load test patterns."""

    CONSTANT = "constant"  # Constant rate
    STEP = "step"  # Step up/down
    RAMP = "ramp"  # Ramp up/down
    SAWTOOTH = "sawtooth"  # Repeated ramp up/down
    SINE = "sine"  # Sinusoidal pattern
    RANDOM = "random"  # Random pattern


class LoadTestResult:
    """
    Results of a load test.

    Attributes:
        name: Test name
        duration: Test duration
        total_requests: Total number of requests
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        latencies: List of latencies
        errors: Dictionary of errors and counts
        timestamps: List of request timestamps
    """

    def __init__(self, name: str, duration: float, target: Dict[str, Any]):
        """
        Initialize load test result.

        Args:
            name: Test name
            duration: Test duration
            target: Target information
        """
        self.name = name
        self.duration = duration
        self.target = target

        # Request statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # Latency statistics
        self.latencies: List[float] = []

        # Error statistics
        self.errors: Dict[str, int] = {}

        # Timestamps
        self.timestamps: List[float] = []

        # Results by time window
        self.window_results: Dict[str, Dict[str, Any]] = {}

        # Start time
        self.start_time = time.time()
        self.end_time = None

    def add_result(
        self,
        timestamp: float,
        success: bool,
        latency: float,
        error: Optional[str] = None,
        window_size: int = 1,
    ) -> None:
        """
        Add a request result.

        Args:
            timestamp: Request timestamp
            success: Whether the request was successful
            latency: Request latency
            error: Error message if request failed
            window_size: Size of time window in seconds
        """
        # Update counters
        self.total_requests += 1

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

            if error:
                if error not in self.errors:
                    self.errors[error] = 0
                self.errors[error] += 1

        # Add latency
        self.latencies.append(latency)

        # Add timestamp
        self.timestamps.append(timestamp)

        # Update window results
        window_key = str(int(timestamp / window_size) * window_size)

        if window_key not in self.window_results:
            self.window_results[window_key] = {
                "timestamp": float(window_key),
                "requests": 0,
                "success": 0,
                "failure": 0,
                "latencies": [],
                "errors": {},
            }

        window = self.window_results[window_key]
        window["requests"] += 1

        if success:
            window["success"] += 1
        else:
            window["failure"] += 1

            if error:
                if error not in window["errors"]:
                    window["errors"][error] = 0
                window["errors"][error] += 1

        window["latencies"].append(latency)

    def complete(self) -> None:
        """Mark the test as complete."""
        self.end_time = time.time()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the load test results.

        Returns:
            Summary dictionary
        """
        # Calculate statistics
        success_rate = self.successful_requests / max(1, self.total_requests) * 100

        latency_stats = {
            "min": min(self.latencies) if self.latencies else 0,
            "max": max(self.latencies) if self.latencies else 0,
            "mean": statistics.mean(self.latencies) if self.latencies else 0,
            "median": statistics.median(self.latencies) if self.latencies else 0,
            "p95": percentile(self.latencies, 95) if self.latencies else 0,
            "p99": percentile(self.latencies, 99) if self.latencies else 0,
        }

        # Get top errors
        top_errors = sorted(self.errors.items(), key=lambda x: x[1], reverse=True)[:5]

        # Calculate throughput
        actual_duration = (self.end_time or time.time()) - self.start_time
        throughput = self.total_requests / max(0.001, actual_duration)

        # Create summary
        summary = {
            "name": self.name,
            "target": self.target,
            "duration": self.duration,
            "actual_duration": actual_duration,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "latency": latency_stats,
            "throughput": throughput,
            "top_errors": top_errors,
        }

        return summary

    def save_to_file(self, filename: str, format: str = "json") -> None:
        """
        Save results to a file.

        Args:
            filename: File to save results to
            format: File format (json or csv)

        Raises:
            ValueError: If format is invalid
        """
        if format == "json":
            # Save as JSON
            with open(filename, "w") as f:
                json.dump(self.get_summary(), f, indent=2)

        elif format == "csv":
            # Save as CSV
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow(["timestamp", "latency", "success", "error"])

                # Write data
                for i in range(len(self.timestamps)):
                    timestamp = self.timestamps[i]
                    latency = self.latencies[i]
                    success = i < self.successful_requests
                    error = ""

                    if not success:
                        for error_msg, count in self.errors.items():
                            if i < count:
                                error = error_msg
                                break

                    writer.writerow([timestamp, latency, success, error])

        else:
            raise ValueError(f"Invalid format: {format}")

    def get_window_results(self, window_size: int = 1) -> List[Dict[str, Any]]:
        """
        Get results by time window.

        Args:
            window_size: Size of time window in seconds

        Returns:
            List of window results
        """
        # Recompute windows if window size doesn't match
        if not self.window_results or len(self.window_results) == 0:
            # Recompute all results
            self.window_results = {}

            for i in range(len(self.timestamps)):
                timestamp = self.timestamps[i]
                latency = self.latencies[i]
                success = i < self.successful_requests
                error = ""

                if not success:
                    for error_msg, count in self.errors.items():
                        if i < count:
                            error = error_msg
                            break

                self.add_result(
                    timestamp=timestamp,
                    success=success,
                    latency=latency,
                    error=error,
                    window_size=window_size,
                )

        # Get sorted window results
        windows = sorted(self.window_results.values(), key=lambda x: x["timestamp"])

        # Calculate additional statistics
        for window in windows:
            if window["latencies"]:
                window["latency_mean"] = statistics.mean(window["latencies"])
                window["latency_p95"] = percentile(window["latencies"], 95)
            else:
                window["latency_mean"] = 0
                window["latency_p95"] = 0

            # Remove raw latencies to save memory
            del window["latencies"]

        return windows


class LoadTestRunner:
    """
    Runner for load tests.

    This class provides methods for running load tests against
    cloud services.
    """

    def __init__(self, num_threads: int = 10, ramp_up_time: float = 0, cooldown_time: float = 0):
        """
        Initialize load test runner.

        Args:
            num_threads: Number of worker threads
            ramp_up_time: Time to ramp up load
            cooldown_time: Time to cool down after test
        """
        self.num_threads = num_threads
        self.ramp_up_time = ramp_up_time
        self.cooldown_time = cooldown_time

        # Thread pool
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix="load_test"
        )

        # Active tests
        self.active_tests: Dict[str, Dict[str, Any]] = {}

        # Thread safety
        self.lock = threading.RLock()

    def _worker_task(
        self,
        test_id: str,
        target_function: Callable,
        args: tuple,
        kwargs: Dict[str, Any],
        rate_limiter: Optional["RateLimiter"] = None,
        result: Optional[LoadTestResult] = None,
        max_requests: Optional[int] = None,
        max_duration: Optional[float] = None,
    ) -> None:
        """
        Worker task for load testing.

        Args:
            test_id: Test ID
            target_function: Function to test
            args: Function arguments
            kwargs: Function keyword arguments
            rate_limiter: Rate limiter
            result: Result object
            max_requests: Maximum number of requests
            max_duration: Maximum test duration
        """
        # Track requests
        request_count = 0
        start_time = time.time()

        # Check if test has been canceled
        with self.lock:
            if test_id not in self.active_tests:
                return

            test = self.active_tests[test_id]
            if test.get("canceled", False):
                return

        # Run test
        while True:
            # Check if test should stop
            if max_requests is not None and request_count >= max_requests:
                break

            if max_duration is not None and time.time() - start_time >= max_duration:
                break

            with self.lock:
                if test_id not in self.active_tests:
                    break

                test = self.active_tests[test_id]
                if test.get("canceled", False):
                    break

            # Wait for rate limiter
            if rate_limiter:
                rate_limiter.acquire()

            # Execute request
            request_start = time.time()
            success = True
            error = None

            try:
                target_function(*args, **kwargs)
            except Exception as e:
                success = False
                error = str(e)

            # Calculate latency
            latency = time.time() - request_start

            # Record result
            if result:
                result.add_result(
                    timestamp=request_start,
                    success=success,
                    latency=latency,
                    error=error,
                )

            # Increment counter
            request_count += 1

    def run_load_test(
        self,
        name: str,
        target_function: Callable,
        args: Optional[tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        target_rps: float = 10.0,
        duration: float = 60.0,
        pattern: LoadPattern = LoadPattern.CONSTANT,
        pattern_params: Optional[Dict[str, Any]] = None,
        max_requests: Optional[int] = None,
    ) -> LoadTestResult:
        """
        Run a load test.

        Args:
            name: Test name
            target_function: Function to test
            args: Function arguments
            kwargs: Function keyword arguments
            target_rps: Target requests per second
            duration: Test duration in seconds
            pattern: Load pattern
            pattern_params: Pattern parameters
            max_requests: Maximum number of requests

        Returns:
            Load test result

        Raises:
            ValueError: If parameters are invalid
        """
        # Normalize parameters
        args = args or ()
        kwargs = kwargs or {}
        pattern_params = pattern_params or {}

        # Create test ID
        import uuid

        test_id = str(uuid.uuid4())

        # Create result object
        result = LoadTestResult(
            name=name,
            duration=duration,
            target={
                "function": target_function.__name__,
                "target_rps": target_rps,
                "pattern": pattern.value,
                "pattern_params": pattern_params,
            },
        )

        # Create rate limiter based on pattern
        if pattern == LoadPattern.CONSTANT:
            rate_limiter = ConstantRateLimiter(target_rps)
        elif pattern == LoadPattern.STEP:
            rate_limiter = StepRateLimiter(
                initial_rps=pattern_params.get("initial_rps", 1.0),
                target_rps=target_rps,
                step_size=pattern_params.get("step_size", 1.0),
                step_duration=pattern_params.get("step_duration", 10.0),
            )
        elif pattern == LoadPattern.RAMP:
            rate_limiter = RampRateLimiter(
                initial_rps=pattern_params.get("initial_rps", 1.0),
                target_rps=target_rps,
                duration=duration,
            )
        elif pattern == LoadPattern.SAWTOOTH:
            rate_limiter = SawtoothRateLimiter(
                min_rps=pattern_params.get("min_rps", 1.0),
                max_rps=target_rps,
                period=pattern_params.get("period", 60.0),
            )
        elif pattern == LoadPattern.SINE:
            rate_limiter = SineRateLimiter(
                base_rps=pattern_params.get("base_rps", target_rps / 2),
                amplitude=pattern_params.get("amplitude", target_rps / 2),
                period=pattern_params.get("period", 60.0),
            )
        elif pattern == LoadPattern.RANDOM:
            rate_limiter = RandomRateLimiter(
                min_rps=pattern_params.get("min_rps", 1.0),
                max_rps=target_rps,
                change_period=pattern_params.get("change_period", 5.0),
            )
        else:
            raise ValueError(f"Invalid pattern: {pattern}")

        # Calculate number of worker threads needed
        worker_count = min(self.num_threads, int(target_rps * 2) + 1)

        # Add test to active tests
        with self.lock:
            self.active_tests[test_id] = {
                "id": test_id,
                "name": name,
                "start_time": time.time(),
                "duration": duration,
                "result": result,
                "canceled": False,
            }

        # Start worker threads
        futures = []
        for _ in range(worker_count):
            future = self.thread_pool.submit(
                self._worker_task,
                test_id,
                target_function,
                args,
                kwargs,
                rate_limiter,
                result,
                max_requests,
                duration,
            )
            futures.append(future)

        # Wait for test to complete
        try:
            # Wait for duration
            time.sleep(duration)

            # Mark test as completed
            with self.lock:
                if test_id in self.active_tests:
                    self.active_tests[test_id]["canceled"] = True

            # Wait for all worker threads to complete
            for future in futures:
                try:
                    future.result(timeout=self.cooldown_time)
                except concurrent.futures.TimeoutError:
                    # Worker didn't finish in time
                    pass
                except Exception as e:
                    logger.error(f"Worker error: {str(e)}")

            # Complete result
            result.complete()

            return result

        finally:
            # Clean up
            with self.lock:
                if test_id in self.active_tests:
                    del self.active_tests[test_id]

    def stop_all_tests(self) -> None:
        """Stop all active tests."""
        with self.lock:
            for test_id, test in self.active_tests.items():
                test["canceled"] = True

    def stop_test(self, test_id: str) -> bool:
        """
        Stop a specific test.

        Args:
            test_id: Test ID

        Returns:
            True if test was stopped, False if not found
        """
        with self.lock:
            if test_id in self.active_tests:
                self.active_tests[test_id]["canceled"] = True
                return True

            return False

    def get_active_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active tests.

        Returns:
            Dictionary of active tests
        """
        with self.lock:
            return {k: v.copy() for k, v in self.active_tests.items()}


class RateLimiter:
    """
    Base class for rate limiters.

    Rate limiters control the rate of requests in load tests.
    """

    def __init__(self):
        """Initialize rate limiter."""
        self.start_time = time.time()

    def acquire(self) -> None:
        """
        Acquire a permit to execute a request.

        This method blocks until a permit is available.
        """
        raise NotImplementedError("Subclasses must implement acquire")

    def get_current_rate(self) -> float:
        """
        Get the current rate limit.

        Returns:
            Current rate limit in requests per second
        """
        raise NotImplementedError("Subclasses must implement get_current_rate")


class ConstantRateLimiter(RateLimiter):
    """
    Rate limiter with a constant rate.

    This rate limiter maintains a constant rate of requests.
    """

    def __init__(self, rps: float):
        """
        Initialize constant rate limiter.

        Args:
            rps: Requests per second
        """
        super().__init__()
        self.rps = rps
        self.interval = 1.0 / max(0.01, rps)
        self.last_request = time.time()
        self.lock = threading.RLock()

    def acquire(self) -> None:
        """
        Acquire a permit to execute a request.

        This method blocks until a permit is available.
        """
        with self.lock:
            # Calculate time until next request
            now = time.time()
            elapsed = now - self.last_request
            wait_time = max(0, self.interval - elapsed)

            # Wait if needed
            if wait_time > 0:
                time.sleep(wait_time)

            # Update last request time
            self.last_request = time.time()

    def get_current_rate(self) -> float:
        """
        Get the current rate limit.

        Returns:
            Current rate limit in requests per second
        """
        return self.rps


class StepRateLimiter(RateLimiter):
    """
    Rate limiter with a step pattern.

    This rate limiter increases the rate in steps.
    """

    def __init__(
        self,
        initial_rps: float,
        target_rps: float,
        step_size: float = 1.0,
        step_duration: float = 10.0,
    ):
        """
        Initialize step rate limiter.

        Args:
            initial_rps: Initial requests per second
            target_rps: Target requests per second
            step_size: Size of each step
            step_duration: Duration of each step
        """
        super().__init__()
        self.initial_rps = initial_rps
        self.target_rps = target_rps
        self.step_size = step_size
        self.step_duration = step_duration
        self.lock = threading.RLock()

        # Internal rate limiter
        self.rate_limiter = ConstantRateLimiter(initial_rps)

    def acquire(self) -> None:
        """
        Acquire a permit to execute a request.

        This method blocks until a permit is available.
        """
        with self.lock:
            # Calculate current RPS based on time
            elapsed = time.time() - self.start_time
            step = int(elapsed / self.step_duration)
            current_rps = min(self.target_rps, self.initial_rps + step * self.step_size)

            # Update rate limiter if needed
            if current_rps != self.rate_limiter.rps:
                self.rate_limiter = ConstantRateLimiter(current_rps)

        # Acquire from internal rate limiter
        self.rate_limiter.acquire()

    def get_current_rate(self) -> float:
        """
        Get the current rate limit.

        Returns:
            Current rate limit in requests per second
        """
        elapsed = time.time() - self.start_time
        step = int(elapsed / self.step_duration)
        current_rps = min(self.target_rps, self.initial_rps + step * self.step_size)

        return current_rps


class RampRateLimiter(RateLimiter):
    """
    Rate limiter with a ramp pattern.

    This rate limiter increases the rate linearly.
    """

    def __init__(self, initial_rps: float, target_rps: float, duration: float):
        """
        Initialize ramp rate limiter.

        Args:
            initial_rps: Initial requests per second
            target_rps: Target requests per second
            duration: Ramp duration
        """
        super().__init__()
        self.initial_rps = initial_rps
        self.target_rps = target_rps
        self.duration = duration
        self.rate_range = target_rps - initial_rps
        self.lock = threading.RLock()

        # Last request time
        self.last_request = time.time()

    def acquire(self) -> None:
        """
        Acquire a permit to execute a request.

        This method blocks until a permit is available.
        """
        with self.lock:
            # Calculate current RPS based on time
            elapsed = time.time() - self.start_time
            progress = min(1.0, elapsed / self.duration)
            current_rps = self.initial_rps + progress * self.rate_range

            # Calculate interval
            interval = 1.0 / max(0.01, current_rps)

            # Calculate time until next request
            now = time.time()
            wait_time = max(0, interval - (now - self.last_request))

            # Wait if needed
            if wait_time > 0:
                time.sleep(wait_time)

            # Update last request time
            self.last_request = time.time()

    def get_current_rate(self) -> float:
        """
        Get the current rate limit.

        Returns:
            Current rate limit in requests per second
        """
        elapsed = time.time() - self.start_time
        progress = min(1.0, elapsed / self.duration)
        current_rps = self.initial_rps + progress * self.rate_range

        return current_rps


class SawtoothRateLimiter(RateLimiter):
    """
    Rate limiter with a sawtooth pattern.

    This rate limiter repeatedly ramps up and down.
    """

    def __init__(self, min_rps: float, max_rps: float, period: float = 60.0):
        """
        Initialize sawtooth rate limiter.

        Args:
            min_rps: Minimum requests per second
            max_rps: Maximum requests per second
            period: Period of one cycle
        """
        super().__init__()
        self.min_rps = min_rps
        self.max_rps = max_rps
        self.period = period
        self.rate_range = max_rps - min_rps
        self.lock = threading.RLock()

        # Last request time
        self.last_request = time.time()

    def acquire(self) -> None:
        """
        Acquire a permit to execute a request.

        This method blocks until a permit is available.
        """
        with self.lock:
            # Calculate current RPS based on time
            elapsed = time.time() - self.start_time
            cycle_time = elapsed % self.period
            cycle_progress = cycle_time / self.period

            # Calculate current RPS based on sawtooth pattern
            if cycle_progress < 0.5:
                # Ramp up
                progress = cycle_progress * 2
                current_rps = self.min_rps + progress * self.rate_range
            else:
                # Ramp down
                progress = (cycle_progress - 0.5) * 2
                current_rps = self.max_rps - progress * self.rate_range

            # Calculate interval
            interval = 1.0 / max(0.01, current_rps)

            # Calculate time until next request
            now = time.time()
            wait_time = max(0, interval - (now - self.last_request))

            # Wait if needed
            if wait_time > 0:
                time.sleep(wait_time)

            # Update last request time
            self.last_request = time.time()

    def get_current_rate(self) -> float:
        """
        Get the current rate limit.

        Returns:
            Current rate limit in requests per second
        """
        elapsed = time.time() - self.start_time
        cycle_time = elapsed % self.period
        cycle_progress = cycle_time / self.period

        # Calculate current RPS based on sawtooth pattern
        if cycle_progress < 0.5:
            # Ramp up
            progress = cycle_progress * 2
            current_rps = self.min_rps + progress * self.rate_range
        else:
            # Ramp down
            progress = (cycle_progress - 0.5) * 2
            current_rps = self.max_rps - progress * self.rate_range

        return current_rps


class SineRateLimiter(RateLimiter):
    """
    Rate limiter with a sine pattern.

    This rate limiter varies the rate sinusoidally.
    """

    def __init__(self, base_rps: float, amplitude: float, period: float = 60.0):
        """
        Initialize sine rate limiter.

        Args:
            base_rps: Base requests per second
            amplitude: Amplitude of variation
            period: Period of one cycle
        """
        super().__init__()
        self.base_rps = base_rps
        self.amplitude = amplitude
        self.period = period
        self.lock = threading.RLock()

        # Last request time
        self.last_request = time.time()

    def acquire(self) -> None:
        """
        Acquire a permit to execute a request.

        This method blocks until a permit is available.
        """
        with self.lock:
            # Calculate current RPS based on time
            elapsed = time.time() - self.start_time
            cycle_progress = (elapsed % self.period) / self.period

            # Calculate current RPS based on sine pattern
            import math

            current_rps = self.base_rps + self.amplitude * math.sin(cycle_progress * 2 * math.pi)

            # Calculate interval
            interval = 1.0 / max(0.01, current_rps)

            # Calculate time until next request
            now = time.time()
            wait_time = max(0, interval - (now - self.last_request))

            # Wait if needed
            if wait_time > 0:
                time.sleep(wait_time)

            # Update last request time
            self.last_request = time.time()

    def get_current_rate(self) -> float:
        """
        Get the current rate limit.

        Returns:
            Current rate limit in requests per second
        """
        elapsed = time.time() - self.start_time
        cycle_progress = (elapsed % self.period) / self.period

        # Calculate current RPS based on sine pattern
        import math

        current_rps = self.base_rps + self.amplitude * math.sin(cycle_progress * 2 * math.pi)

        return max(0.01, current_rps)


class RandomRateLimiter(RateLimiter):
    """
    Rate limiter with a random pattern.

    This rate limiter varies the rate randomly.
    """

    def __init__(self, min_rps: float, max_rps: float, change_period: float = 5.0):
        """
        Initialize random rate limiter.

        Args:
            min_rps: Minimum requests per second
            max_rps: Maximum requests per second
            change_period: Time between rate changes
        """
        super().__init__()
        self.min_rps = min_rps
        self.max_rps = max_rps
        self.change_period = change_period
        self.lock = threading.RLock()

        # Current RPS and last change time
        # Using random for load testing simulation only, not for security purposes
        self.current_rps = random.uniform(min_rps, max_rps)
        self.last_change = time.time()

        # Last request time
        self.last_request = time.time()

    def acquire(self) -> None:
        """
        Acquire a permit to execute a request.

        This method blocks until a permit is available.
        """
        with self.lock:
            # Check if it's time to change the rate
            now = time.time()
            if now - self.last_change >= self.change_period:
                self.current_rps = random.uniform(self.min_rps, self.max_rps)
                self.last_change = now

            # Calculate interval
            interval = 1.0 / max(0.01, self.current_rps)

            # Calculate time until next request
            wait_time = max(0, interval - (now - self.last_request))

            # Wait if needed
            if wait_time > 0:
                time.sleep(wait_time)

            # Update last request time
            self.last_request = time.time()

    def get_current_rate(self) -> float:
        """
        Get the current rate limit.

        Returns:
            Current rate limit in requests per second
        """
        # Check if it's time to change the rate
        now = time.time()
        if now - self.last_change >= self.change_period:
            with self.lock:
                if now - self.last_change >= self.change_period:
                    self.current_rps = random.uniform(self.min_rps, self.max_rps)
                    self.last_change = now

        return self.current_rps


def percentile(data: List[float], percentile: float) -> float:
    """
    Calculate percentile of a list of values.

    Args:
        data: List of values
        percentile: Percentile to calculate (0-100)

    Returns:
        Percentile value
    """
    if not data:
        return 0.0

    # Sort data
    sorted_data = sorted(data)

    # Calculate index
    index = (len(sorted_data) - 1) * percentile / 100

    # Get percentile
    if index.is_integer():
        return sorted_data[int(index)]
    else:
        lower = int(index)
        upper = lower + 1
        weight = index - lower

        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


def run_load_test(
    target_function: Callable,
    args: Optional[tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    target_rps: float = 10.0,
    duration: float = 60.0,
    pattern: Union[LoadPattern, str] = LoadPattern.CONSTANT,
    pattern_params: Optional[Dict[str, Any]] = None,
    num_threads: int = 10,
) -> LoadTestResult:
    """
    Run a load test (convenience function).

    Args:
        target_function: Function to test
        args: Function arguments
        kwargs: Function keyword arguments
        target_rps: Target requests per second
        duration: Test duration in seconds
        pattern: Load pattern
        pattern_params: Pattern parameters
        num_threads: Number of worker threads

    Returns:
        Load test result
    """
    # Normalize pattern
    if isinstance(pattern, str):
        pattern = LoadPattern(pattern)

    # Create runner
    runner = LoadTestRunner(num_threads=num_threads)

    # Run test
    return runner.run_load_test(
        name=f"Load test for {target_function.__name__}",
        target_function=target_function,
        args=args,
        kwargs=kwargs,
        target_rps=target_rps,
        duration=duration,
        pattern=pattern,
        pattern_params=pattern_params,
    )


def run_benchmark(
    target_function: Callable,
    args: Optional[tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    duration: float = 10.0,
    num_threads: int = 1,
) -> Dict[str, Any]:
    """
    Run a performance benchmark.

    This function runs a load test with a single thread
    to benchmark the performance of a function.

    Args:
        target_function: Function to test
        args: Function arguments
        kwargs: Function keyword arguments
        duration: Test duration in seconds
        num_threads: Number of threads

    Returns:
        Benchmark results
    """
    # Create runner with a high number of threads
    # to ensure we're not limited by the runner
    runner = LoadTestRunner(num_threads=num_threads)

    # Run test with a high target RPS
    # to ensure we're testing the function's maximum performance
    result = runner.run_load_test(
        name=f"Benchmark for {target_function.__name__}",
        target_function=target_function,
        args=args,
        kwargs=kwargs,
        target_rps=1000000.0,  # Very high to ensure no rate limiting
        duration=duration,
        pattern=LoadPattern.CONSTANT,
    )

    # Get summary
    summary = result.get_summary()

    # Create benchmark results
    benchmark = {
        "function": target_function.__name__,
        "duration": duration,
        "threads": num_threads,
        "requests": summary["total_requests"],
        "throughput": summary["throughput"],
        "latency_mean": summary["latency"]["mean"],
        "latency_p95": summary["latency"]["p95"],
        "success_rate": summary["success_rate"],
    }

    return benchmark
