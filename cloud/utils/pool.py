"""
Connection pooling utilities for cloud services.

This module provides tools for efficiently managing connections to cloud services,
reducing resource usage and improving performance.

Copyright (c) 2025 Vikas Sahani
GitHub: https://github.com/VIKAS9793
Email: vikassahani17@gmail.com

Licensed under MIT License - see LICENSE file for details
This copyright and license applies only to the original code in this file,
not to any third-party libraries or dependencies used.
"""

import time
import logging
import threading
import weakref
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, List, Tuple, Union
from enum import Enum
import random
from queue import Queue, Empty
import functools

from cloud.config import CloudProvider
from cloud.errors import CloudTimeoutError, CloudServiceUnavailableError

logger = logging.getLogger(__name__)

# Type variable for connection type
T = TypeVar('T')


class PoolStatus(Enum):
    """Status of a connection pool."""
    ACTIVE = "active"
    DRAINING = "draining"  # Not accepting new connections, but existing ones can complete
    CLOSED = "closed"


class PooledConnection(Generic[T]):
    """
    A wrapper for a connection in the pool.
    
    Attributes:
        connection: The actual connection object
        created_at: Timestamp when connection was created
        last_used: Timestamp when connection was last used
        use_count: Number of times this connection has been used
        is_in_use: Whether the connection is currently being used
    """
    
    def __init__(self, connection: T, validator: Optional[Callable[[T], bool]] = None):
        """
        Initialize a pooled connection.
        
        Args:
            connection: The actual connection object
            validator: Function to validate if connection is still valid
        """
        self.connection = connection
        self.created_at = time.time()
        self.last_used = time.time()
        self.use_count = 0
        self.is_in_use = False
        self.validator = validator
    
    def acquire(self) -> T:
        """
        Mark connection as in use and return it.
        
        Returns:
            The connection
        """
        self.is_in_use = True
        self.last_used = time.time()
        self.use_count += 1
        return self.connection
    
    def release(self) -> None:
        """Release the connection back to the pool."""
        self.is_in_use = False
        self.last_used = time.time()
    
    def is_valid(self) -> bool:
        """
        Check if the connection is still valid.
        
        Returns:
            True if connection is valid, False otherwise
        """
        if self.validator:
            try:
                return self.validator(self.connection)
            except Exception as e:
                logger.warning(f"Connection validation failed: {str(e)}")
                return False
        return True
    
    def close(self) -> None:
        """Close the underlying connection if it has a close method."""
        if hasattr(self.connection, 'close') and callable(getattr(self.connection, 'close')):
            try:
                self.connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {str(e)}")


class ConnectionPool(Generic[T]):
    """
    A pool of connections to a cloud service.
    
    This class manages a pool of connections to a specific cloud service,
    allowing for connection reuse and limiting the number of connections.
    """
    
    def __init__(
        self,
        factory: Callable[[], T],
        validator: Optional[Callable[[T], bool]] = None,
        max_size: int = 10,
        min_size: int = 0,
        max_age: float = 600.0,
        max_idle: float = 300.0,
        max_uses: int = 100,
        block_timeout: float = 30.0,
    ):
        """
        Initialize a connection pool.
        
        Args:
            factory: Function to create new connections
            validator: Function to validate existing connections
            max_size: Maximum number of connections in the pool
            min_size: Minimum number of connections to keep in the pool
            max_age: Maximum age of a connection in seconds
            max_idle: Maximum idle time for a connection in seconds
            max_uses: Maximum number of times a connection can be used
            block_timeout: Timeout for blocking operations in seconds
        """
        self.factory = factory
        self.validator = validator
        self.max_size = max_size
        self.min_size = min_size
        self.max_age = max_age
        self.max_idle = max_idle
        self.max_uses = max_uses
        self.block_timeout = block_timeout
        
        # Pool state
        self.status = PoolStatus.ACTIVE
        self.available: Queue[PooledConnection[T]] = Queue()
        self.in_use: Dict[int, PooledConnection[T]] = {}
        self.size = 0
        
        # Maintenance variables
        self.last_maintenance = time.time()
        self.maintenance_interval = 60.0  # seconds
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize pool with min_size connections
        self._initialize_pool()
        
        # Start maintenance thread
        self._start_maintenance_thread()
    
    def _initialize_pool(self) -> None:
        """Initialize the pool with min_size connections."""
        with self.lock:
            for _ in range(self.min_size):
                try:
                    conn = self._create_connection()
                    self.available.put(conn)
                except Exception as e:
                    logger.error(f"Error initializing connection pool: {str(e)}")
    
    def _create_connection(self) -> PooledConnection[T]:
        """
        Create a new connection.
        
        Returns:
            A new pooled connection
            
        Raises:
            Exception: If connection creation fails
        """
        connection = self.factory()
        pooled_conn = PooledConnection(connection, self.validator)
        self.size += 1
        return pooled_conn
    
    def _start_maintenance_thread(self) -> None:
        """Start the maintenance thread."""
        def maintenance_task():
            while self.status != PoolStatus.CLOSED:
                time.sleep(self.maintenance_interval)
                try:
                    self._perform_maintenance()
                except Exception as e:
                    logger.error(f"Error in connection pool maintenance: {str(e)}")
        
        thread = threading.Thread(target=maintenance_task, daemon=True)
        thread.start()
    
    def _perform_maintenance(self) -> None:
        """
        Perform maintenance on the pool.
        
        This method removes invalid, old, or excessively used connections
        and creates new ones if the pool is below min_size.
        """
        with self.lock:
            if self.status == PoolStatus.CLOSED:
                return
            
            current_time = time.time()
            self.last_maintenance = current_time
            
            # Collect connections to check
            to_check = []
            while not self.available.empty():
                try:
                    conn = self.available.get_nowait()
                    to_check.append(conn)
                except Empty:
                    break
            
            # Check and filter connections
            valid_connections = []
            for conn in to_check:
                should_close = False
                
                # Check if connection is too old
                if current_time - conn.created_at > self.max_age:
                    should_close = True
                
                # Check if connection has been idle too long
                elif current_time - conn.last_used > self.max_idle:
                    should_close = True
                
                # Check if connection has been used too many times
                elif self.max_uses > 0 and conn.use_count >= self.max_uses:
                    should_close = True
                
                # Check if connection is still valid
                elif not conn.is_valid():
                    should_close = True
                
                if should_close:
                    conn.close()
                    self.size -= 1
                else:
                    valid_connections.append(conn)
            
            # Return valid connections to the pool
            for conn in valid_connections:
                self.available.put(conn)
            
            # If we're active and below min_size, create new connections
            if self.status == PoolStatus.ACTIVE and self.size < self.min_size:
                for _ in range(min(self.min_size - self.size, self.max_size - self.size)):
                    try:
                        conn = self._create_connection()
                        self.available.put(conn)
                    except Exception as e:
                        logger.error(f"Error creating connection during maintenance: {str(e)}")
    
    def acquire(self) -> T:
        """
        Acquire a connection from the pool.
        
        Returns:
            A connection
            
        Raises:
            CloudTimeoutError: If no connection available within timeout
            CloudServiceUnavailableError: If pool is closed
        """
        if self.status == PoolStatus.CLOSED:
            raise CloudServiceUnavailableError("Connection pool is closed")
        
        # First try to get a connection from the available pool
        try:
            conn = self.available.get(block=False)
            
            # Validate the connection
            if not conn.is_valid():
                conn.close()
                self.size -= 1
                # Continue to create a new connection
            else:
                # Connection is valid, use it
                with self.lock:
                    conn_id = id(conn.connection)
                    self.in_use[conn_id] = conn
                return conn.acquire()
            
        except Empty:
            # No connection available in the pool
            pass
        
        # Try to create a new connection if below max_size
        with self.lock:
            if self.size < self.max_size:
                try:
                    conn = self._create_connection()
                    conn_id = id(conn.connection)
                    self.in_use[conn_id] = conn
                    return conn.acquire()
                except Exception as e:
                    logger.error(f"Error creating new connection: {str(e)}")
                    # Fall back to waiting for an available connection
            
            # If we reached max_size or couldn't create a new connection,
            # wait for an available connection
            if self.status == PoolStatus.DRAINING:
                raise CloudServiceUnavailableError("Connection pool is draining")
        
        try:
            start_time = time.time()
            while time.time() - start_time < self.block_timeout:
                try:
                    conn = self.available.get(block=True, timeout=1.0)
                    
                    # Validate the connection
                    if not conn.is_valid():
                        with self.lock:
                            conn.close()
                            self.size -= 1
                        # Continue waiting for a valid connection
                        continue
                    
                    # Connection is valid, use it
                    with self.lock:
                        conn_id = id(conn.connection)
                        self.in_use[conn_id] = conn
                    return conn.acquire()
                    
                except Empty:
                    # Timeout on queue get, check if we should continue waiting
                    if time.time() - start_time >= self.block_timeout:
                        break
                    continue
            
            # If we get here, we've timed out
            raise CloudTimeoutError("Timeout waiting for available connection")
            
        except Exception as e:
            if isinstance(e, CloudTimeoutError):
                raise
            logger.error(f"Error acquiring connection: {str(e)}")
            raise CloudServiceUnavailableError(f"Error acquiring connection: {str(e)}")
    
    def release(self, connection: T) -> None:
        """
        Release a connection back to the pool.
        
        Args:
            connection: The connection to release
        """
        with self.lock:
            conn_id = id(connection)
            if conn_id in self.in_use:
                pooled_conn = self.in_use.pop(conn_id)
                pooled_conn.release()
                
                # If pool is not closed and connection is still valid, return it to the pool
                if self.status != PoolStatus.CLOSED and pooled_conn.is_valid():
                    # Check if connection should be retired
                    current_time = time.time()
                    if (current_time - pooled_conn.created_at > self.max_age or
                            self.max_uses > 0 and pooled_conn.use_count >= self.max_uses):
                        pooled_conn.close()
                        self.size -= 1
                    else:
                        self.available.put(pooled_conn)
                else:
                    # Connection is invalid or pool is closed, don't return to pool
                    pooled_conn.close()
                    self.size -= 1
            else:
                logger.warning(f"Attempted to release a connection not managed by this pool")
    
    def drain(self, timeout: Optional[float] = None) -> None:
        """
        Drain the connection pool.
        
        This stops accepting new connections but allows existing ones to complete.
        
        Args:
            timeout: Timeout for waiting for connections to be released
        """
        with self.lock:
            self.status = PoolStatus.DRAINING
        
        if timeout is not None:
            start_time = time.time()
            while self.in_use and time.time() - start_time < timeout:
                time.sleep(0.1)
    
    def close(self) -> None:
        """
        Close the connection pool.
        
        This closes all connections and prevents the pool from being used.
        """
        with self.lock:
            self.status = PoolStatus.CLOSED
            
            # Close all available connections
            while not self.available.empty():
                try:
                    conn = self.available.get_nowait()
                    conn.close()
                except Empty:
                    break
            
            # Close all in-use connections
            for conn in list(self.in_use.values()):
                conn.close()
            
            self.in_use.clear()
            self.size = 0
    
    def __enter__(self) -> 'ConnectionPool[T]':
        """Context manager enter."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


class PoolManager:
    """
    Manager for multiple connection pools.
    
    This class manages connection pools for different cloud providers and services.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(PoolManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the pool manager (only once for singleton)."""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # Pools storage: {provider: {service: {name: pool}}}
        self.pools: Dict[str, Dict[str, Dict[str, ConnectionPool]]] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        self._initialized = True
    
    def get_pool(
        self,
        provider: Union[CloudProvider, str],
        service: str,
        name: str = "default",
        factory: Optional[Callable[[], T]] = None,
        **pool_kwargs
    ) -> ConnectionPool[T]:
        """
        Get or create a connection pool.
        
        Args:
            provider: Cloud provider
            service: Service name
            name: Pool name (for multiple pools for the same service)
            factory: Connection factory function
            **pool_kwargs: Additional parameters for ConnectionPool
            
        Returns:
            A connection pool
            
        Raises:
            ValueError: If factory is None and pool doesn't exist
        """
        # Normalize provider
        if isinstance(provider, CloudProvider):
            provider_str = provider.value
        else:
            provider_str = str(provider).lower()
        
        # Check if pool exists
        with self.lock:
            # Get pools for this provider
            provider_pools = self.pools.setdefault(provider_str, {})
            
            # Get pools for this service
            service_pools = provider_pools.setdefault(service, {})
            
            # Check if the pool exists
            if name in service_pools:
                return service_pools[name]
            
            # Create a new pool if factory is provided
            if factory is None:
                raise ValueError(f"Factory must be provided to create a new pool for {provider_str}/{service}/{name}")
            
            pool = ConnectionPool(factory, **pool_kwargs)
            service_pools[name] = pool
            return pool
    
    def close_pool(
        self,
        provider: Union[CloudProvider, str],
        service: str,
        name: str = "default"
    ) -> None:
        """
        Close a specific connection pool.
        
        Args:
            provider: Cloud provider
            service: Service name
            name: Pool name
        """
        # Normalize provider
        if isinstance(provider, CloudProvider):
            provider_str = provider.value
        else:
            provider_str = str(provider).lower()
        
        with self.lock:
            # Check if provider exists
            if provider_str not in self.pools:
                return
            
            # Check if service exists
            if service not in self.pools[provider_str]:
                return
            
            # Check if pool exists
            if name not in self.pools[provider_str][service]:
                return
            
            # Close the pool
            self.pools[provider_str][service][name].close()
            
            # Remove pool from manager
            del self.pools[provider_str][service][name]
            
            # Clean up empty dictionaries
            if not self.pools[provider_str][service]:
                del self.pools[provider_str][service]
            if not self.pools[provider_str]:
                del self.pools[provider_str]
    
    def close_all_pools(self) -> None:
        """Close all connection pools."""
        with self.lock:
            for provider in list(self.pools.keys()):
                for service in list(self.pools[provider].keys()):
                    for name in list(self.pools[provider][service].keys()):
                        self.pools[provider][service][name].close()
            
            # Clear all pools
            self.pools = {}


# Global pool manager instance
_pool_manager = PoolManager()

def get_pool_manager() -> PoolManager:
    """Get the global pool manager instance."""
    return _pool_manager


def connection_pooled(
    provider: Union[CloudProvider, str],
    service: str,
    pool_name: str = "default"
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to use connection pooling for a function.
    
    This decorator transforms a function that creates and uses a connection
    into one that gets a connection from a pool and releases it back when done.
    
    The decorated function must have a parameter named 'connection_factory'
    that is a function creating the connection.
    
    Args:
        provider: Cloud provider
        service: Service name
        pool_name: Pool name
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get connection factory from kwargs
            factory = kwargs.pop('connection_factory', None)
            if factory is None:
                raise ValueError("connection_factory must be provided")
            
            # Get pool
            pool = _pool_manager.get_pool(
                provider=provider,
                service=service,
                name=pool_name,
                factory=factory,
                **kwargs.get('pool_kwargs', {})
            )
            
            # Remove pool_kwargs if present
            if 'pool_kwargs' in kwargs:
                del kwargs['pool_kwargs']
            
            # Acquire connection
            connection = pool.acquire()
            
            try:
                # Call original function with the connection
                kwargs['connection'] = connection
                return func(*args, **kwargs)
            finally:
                # Release connection back to pool
                pool.release(connection)
        
        return wrapper
    
    return decorator 