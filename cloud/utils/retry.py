"""
Retry utilities for cloud service operations.

This module provides decorators and utilities for implementing
retry strategies when interacting with cloud services.
"""

import functools
import logging
import secrets
import time
from enum import Enum
from typing import Callable, List, Optional, Type, TypeVar, Union

from cloud.errors import (
    CloudError,
    CloudNetworkError,
    CloudQuotaExceededError,
    CloudServiceUnavailableError,
    CloudTimeoutError,
)

logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar("T")


class RetryStrategy(Enum):
    """Retry backoff strategies."""

    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    RANDOM = "random"


class RetryPolicy:
    """
    Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        strategy: Backoff strategy
        retry_on_exceptions: Exception types to retry on
        jitter: Add randomness to delay timing (0-1, fraction of delay)
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_delay: float = 60.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        retry_on_exceptions: Optional[List[Type[Exception]]] = None,
        jitter: float = 0.1,
    ):
        """
        Initialize retry policy.

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            strategy: Backoff strategy for calculating delay
            retry_on_exceptions: Exception types to retry on
            jitter: Add randomness to delay timing (0-1, fraction of delay)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.jitter = min(max(jitter, 0.0), 1.0)  # Clamp to 0-1 range

        # Default retryable exceptions if none provided
        if retry_on_exceptions is None:
            self.retry_on_exceptions = [
                CloudNetworkError,
                CloudServiceUnavailableError,
                CloudTimeoutError,
                CloudQuotaExceededError,
            ]
        else:
            self.retry_on_exceptions = retry_on_exceptions

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate the delay for a specific retry attempt.

        Args:
            attempt: Current retry attempt (1-based)

        Returns:
            Delay in seconds
        """
        if attempt < 1:
            return 0

        # Calculate base delay based on strategy
        if self.strategy == RetryStrategy.FIXED:
            delay = self.retry_delay
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.retry_delay * attempt
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.retry_delay * (2 ** (attempt - 1))
        elif self.strategy == RetryStrategy.FIBONACCI:
            # Calculate Fibonacci number (simplified approach)
            a, b = 1, 1
            for _ in range(attempt - 1):
                a, b = b, a + b
            delay = self.retry_delay * a
        elif self.strategy == RetryStrategy.RANDOM:
            delay = secrets.SystemRandom().uniform(self.retry_delay, self.retry_delay * attempt)
        else:
            delay = self.retry_delay

        # Apply jitter if configured
        if self.jitter > 0:
            jitter_amount = delay * self.jitter
            delay = secrets.SystemRandom().uniform(delay - jitter_amount, delay + jitter_amount)

        # Cap at max delay
        return min(delay, self.max_delay)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if an operation should be retried.

        Args:
            exception: The exception that occurred
            attempt: Current retry attempt

        Returns:
            True if should retry, False otherwise
        """
        # Check if max retries reached
        if attempt >= self.max_retries:
            return False

        # Check if exception is retryable
        for exception_type in self.retry_on_exceptions:
            if isinstance(exception, exception_type):
                return True

        return False


def retry(
    max_retries: Optional[int] = None,
    retry_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    strategy: Optional[Union[RetryStrategy, str]] = None,
    retry_on_exceptions: Optional[List[Type[Exception]]] = None,
    jitter: Optional[float] = None,
    policy: Optional[RetryPolicy] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry cloud operations with configurable backoff.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        strategy: Backoff strategy for calculating delay
        retry_on_exceptions: Exception types to retry on
        jitter: Add randomness to delay timing (0-1, fraction of delay)
        policy: Predefined retry policy to use

    Returns:
        Decorator function
    """
    # Create policy from parameters or use provided policy
    if policy is None:
        policy_args = {}
        if max_retries is not None:
            policy_args["max_retries"] = max_retries
        if retry_delay is not None:
            policy_args["retry_delay"] = retry_delay
        if max_delay is not None:
            policy_args["max_delay"] = max_delay
        if retry_on_exceptions is not None:
            policy_args["retry_on_exceptions"] = retry_on_exceptions
        if jitter is not None:
            policy_args["jitter"] = jitter

        if strategy is not None:
            if isinstance(strategy, str):
                try:
                    policy_args["strategy"] = RetryStrategy(strategy.lower())
                except ValueError:
                    logger.warning(f"Unknown retry strategy: {strategy}, using default")
            else:
                policy_args["strategy"] = strategy

        retry_policy = RetryPolicy(**policy_args)
    else:
        retry_policy = policy

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            attempt = 0

            while True:
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    attempt += 1

                    # Determine if retryable
                    should_retry = retry_policy.should_retry(e, attempt)

                    if not should_retry:
                        # Log final failure if it's a CloudError
                        if isinstance(e, CloudError):
                            e.log(logging.ERROR)
                        raise

                    # Calculate delay
                    delay = retry_policy.calculate_delay(attempt)

                    # Log retry attempt
                    operation = func.__name__
                    logger.warning(
                        f"Retrying {operation} (attempt {attempt}/{retry_policy.max_retries}) "
                        f"after {delay:.2f}s due to: {type(e).__name__}: {str(e)}"
                    )

                    # Wait before retrying
                    time.sleep(delay)

        return wrapper

    return decorator


# Common retry policies
DEFAULT_POLICY = RetryPolicy()

AGGRESSIVE_POLICY = RetryPolicy(
    max_retries=5,
    retry_delay=1.0,
    max_delay=120.0,
    strategy=RetryStrategy.EXPONENTIAL,
    jitter=0.2,
)

CONSERVATIVE_POLICY = RetryPolicy(
    max_retries=2,
    retry_delay=2.0,
    max_delay=30.0,
    strategy=RetryStrategy.LINEAR,
    jitter=0.1,
)


# Convenience decorators with predefined policies
def retry_default(func: Callable[..., T]) -> Callable[..., T]:
    """Retry with default policy."""
    return retry(policy=DEFAULT_POLICY)(func)


def retry_aggressive(func: Callable[..., T]) -> Callable[..., T]:
    """Retry with aggressive policy (more retries, exponential backoff)."""
    return retry(policy=AGGRESSIVE_POLICY)(func)


def retry_conservative(func: Callable[..., T]) -> Callable[..., T]:
    """Retry with conservative policy (fewer retries, linear backoff)."""
    return retry(policy=CONSERVATIVE_POLICY)(func)


# Circuit breaker pattern implementation
class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests flow through
    OPEN = "open"  # Failure threshold exceeded, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreaker:
    """
    Implementation of the circuit breaker pattern.

    Circuit breaker prevents calling failing operations repeatedly,
    which can cascade into system-wide failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        exception_types: Optional[List[Type[Exception]]] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds to wait before attempting recovery
            exception_types: Exception types that count as failures
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        if exception_types is None:
            self.exception_types = [Exception]
        else:
            self.exception_types = exception_types

        # State variables
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function return value

        Raises:
            Exception: Original exception if circuit is closed
            CircuitBreakerError: If circuit is open
        """
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.info("Circuit breaker transitioning to half-open state")
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                # Fail fast
                logger.warning("Circuit breaker open, failing fast")
                raise CloudServiceUnavailableError(
                    "Circuit breaker open, service unavailable",
                    details={
                        "failure_count": self.failure_count,
                        "recovery_timeout": self.recovery_timeout,
                        "remaining_timeout": self.recovery_timeout
                        - (time.time() - self.last_failure_time),
                    },
                )

        try:
            result = func(*args, **kwargs)

            # Reset on success if in half-open state
            if self.state == CircuitBreakerState.HALF_OPEN:
                logger.info("Circuit breaker reset to closed state")
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0

            return result

        except Exception as e:
            # Only count whitelisted exceptions
            is_counted_exception = False
            for exception_type in self.exception_types:
                if isinstance(e, exception_type):
                    is_counted_exception = True
                    break

            if is_counted_exception:
                self.failure_count += 1
                self.last_failure_time = time.time()

                # Check if threshold reached
                if (
                    self.state == CircuitBreakerState.CLOSED
                    and self.failure_count >= self.failure_threshold
                ):
                    logger.warning(
                        f"Circuit breaker threshold reached ({self.failure_count} failures)"
                    )
                    self.state = CircuitBreakerState.OPEN

                # If already in half-open state, go back to open
                if self.state == CircuitBreakerState.HALF_OPEN:
                    logger.warning("Circuit breaker returning to open state after failed recovery")
                    self.state = CircuitBreakerState.OPEN

            # Re-raise the original exception
            raise


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    exception_types: Optional[List[Type[Exception]]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to apply circuit breaker pattern to a function.

    Args:
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Time in seconds to wait before attempting recovery
        exception_types: Exception types that count as failures

    Returns:
        Decorator function
    """
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        exception_types=exception_types,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return breaker.execute(func, *args, **kwargs)

        return wrapper

    return decorator


# Combined retry and circuit breaker pattern
def resilient(
    retry_policy: Optional[RetryPolicy] = None,
    circuit_failure_threshold: int = 5,
    circuit_recovery_timeout: float = 60.0,
    circuit_exception_types: Optional[List[Type[Exception]]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to apply both retry and circuit breaker patterns.

    This combines both patterns for maximum resilience.

    Args:
        retry_policy: Retry policy configuration
        circuit_failure_threshold: Circuit breaker failure threshold
        circuit_recovery_timeout: Circuit breaker recovery timeout
        circuit_exception_types: Circuit breaker exception types

    Returns:
        Decorator function
    """
    retry_decorator = retry(policy=retry_policy or DEFAULT_POLICY)
    circuit_decorator = circuit_breaker(
        failure_threshold=circuit_failure_threshold,
        recovery_timeout=circuit_recovery_timeout,
        exception_types=circuit_exception_types,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Apply retry first (innermost), then circuit breaker
        return circuit_decorator(retry_decorator(func))

    return decorator
