import time
import logging
from typing import Callable, Any, Optional, Type
from functools import wraps
from enum import Enum

logger = logging.getLogger()

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascade failures
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Type[Exception] = Exception):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to apply circuit breaker to function
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Original exception: When function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker moved to HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. "
                    f"Will retry after {self.recovery_timeout} seconds from last failure."
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            logger.warning(f"Circuit breaker recorded failure: {str(e)}")
            raise
        except Exception as e:
            # Don't count unexpected exceptions towards circuit breaker
            logger.error(f"Unexpected exception (not counting towards circuit breaker): {str(e)}")
            raise
    
    def _should_attempt_reset(self) -> bool:
        """
        Check if enough time has passed to attempt circuit reset
        """
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """
        Handle successful function execution
        """
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker reset to CLOSED state after successful call")
        
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
    
    def _on_failure(self) -> None:
        """
        Handle failed function execution
        """
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures. "
                f"Will attempt recovery in {self.recovery_timeout} seconds."
            )
    
    def is_open(self) -> bool:
        """
        Check if circuit breaker is open
        """
        return self.state == CircuitState.OPEN
    
    def is_half_open(self) -> bool:
        """
        Check if circuit breaker is half-open
        """
        return self.state == CircuitState.HALF_OPEN
    
    def is_closed(self) -> bool:
        """
        Check if circuit breaker is closed
        """
        return self.state == CircuitState.CLOSED
    
    def reset(self) -> None:
        """
        Manually reset circuit breaker to closed state
        """
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset to CLOSED state")
    
    def get_state(self) -> dict:
        """
        Get current circuit breaker state information
        """
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time,
            'recovery_timeout': self.recovery_timeout,
            'time_until_retry': max(0, self.recovery_timeout - (time.time() - (self.last_failure_time or 0)))
        }

class CircuitBreakerOpenError(Exception):
    """
    Exception raised when circuit breaker is open
    """
    pass

class MultiServiceCircuitBreaker:
    """
    Manages multiple circuit breakers for different services
    """
    
    def __init__(self):
        self.breakers = {}
    
    def get_breaker(self, service_name: str, **kwargs) -> CircuitBreaker:
        """
        Get or create circuit breaker for a service
        
        Args:
            service_name: Name of the service
            kwargs: Circuit breaker configuration
            
        Returns:
            CircuitBreaker instance
        """
        if service_name not in self.breakers:
            self.breakers[service_name] = CircuitBreaker(**kwargs)
        return self.breakers[service_name]
    
    def get_all_states(self) -> dict:
        """
        Get state information for all circuit breakers
        """
        return {
            service: breaker.get_state() 
            for service, breaker in self.breakers.items()
        }
    
    def reset_all(self) -> None:
        """
        Reset all circuit breakers
        """
        for breaker in self.breakers.values():
            breaker.reset()
        logger.info("All circuit breakers have been reset")
    
    def get_open_circuits(self) -> list:
        """
        Get list of services with open circuit breakers
        """
        return [
            service for service, breaker in self.breakers.items()
            if breaker.is_open()
        ]

# Global instance for easy access across modules
global_circuit_breakers = MultiServiceCircuitBreaker()