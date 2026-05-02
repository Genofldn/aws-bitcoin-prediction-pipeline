import time
import random
import logging
from typing import Callable, Any, Optional, Type, Tuple
from functools import wraps

logger = logging.getLogger()

class RetryHandler:
    """
    Retry handler with exponential backoff and jitter
    """
    
    def __init__(self,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True,
                 retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)):
        """
        Initialize retry handler
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter
            retryable_exceptions: Tuple of exception types that should trigger retry
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to apply retry logic to function
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return self.retry(func, *args, **kwargs)
        return wrapper
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic
        
        Args:
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception encountered after all retries exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                if attempt > 0:
                    delay = self.calculate_delay(attempt - 1)
                    logger.info(f"Retrying function {func.__name__} (attempt {attempt + 1}/{self.max_retries + 1}) after {delay:.2f}s delay")
                    time.sleep(delay)
                
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                
                return result
                
            except self.retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}: {str(e)}")
                else:
                    logger.error(f"Function {func.__name__} failed on final attempt {attempt + 1}: {str(e)}")
                
                continue
            
            except Exception as e:
                # Non-retryable exception
                logger.error(f"Function {func.__name__} failed with non-retryable exception: {str(e)}")
                raise
        
        # All retries exhausted
        raise last_exception
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number
        
        Args:
            attempt: Attempt number (0-based for retries)
            
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add up to 25% random jitter
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount
        
        return delay

class ConditionalRetryHandler(RetryHandler):
    """
    Retry handler with custom retry condition
    """
    
    def __init__(self, retry_condition: Callable[[Exception], bool], **kwargs):
        """
        Initialize conditional retry handler
        
        Args:
            retry_condition: Function that takes exception and returns bool indicating if retry should occur
            kwargs: Additional arguments for base RetryHandler
        """
        super().__init__(**kwargs)
        self.retry_condition = retry_condition
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with conditional retry logic
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self.calculate_delay(attempt - 1)
                    logger.info(f"Retrying function {func.__name__} (attempt {attempt + 1}/{self.max_retries + 1}) after {delay:.2f}s delay")
                    time.sleep(delay)
                
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not self.retry_condition(e):
                    logger.error(f"Function {func.__name__} failed with non-retryable exception: {str(e)}")
                    raise
                
                if attempt < self.max_retries:
                    logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}: {str(e)}")
                else:
                    logger.error(f"Function {func.__name__} failed on final attempt {attempt + 1}: {str(e)}")
                
                continue
        
        raise last_exception

class HTTPRetryHandler(RetryHandler):
    """
    Specialized retry handler for HTTP requests
    """
    
    def __init__(self, **kwargs):
        # Default retryable HTTP status codes
        self.retryable_status_codes = {429, 500, 502, 503, 504}
        super().__init__(**kwargs)
    
    def should_retry_http_error(self, exception: Exception) -> bool:
        """
        Determine if HTTP error should be retried
        
        Args:
            exception: Exception to check
            
        Returns:
            bool: True if should retry
        """
        # Check for requests library exceptions
        if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
            return exception.response.status_code in self.retryable_status_codes
        
        # Check for urllib3 exceptions
        if 'timeout' in str(exception).lower():
            return True
        
        if 'connection' in str(exception).lower():
            return True
        
        return False

def retry_on_condition(condition: Callable[[Exception], bool], 
                      max_retries: int = 3,
                      base_delay: float = 1.0,
                      **kwargs) -> Callable:
    """
    Decorator factory for conditional retry
    
    Args:
        condition: Function that takes exception and returns bool
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries
        kwargs: Additional retry handler arguments
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        handler = ConditionalRetryHandler(
            retry_condition=condition,
            max_retries=max_retries,
            base_delay=base_delay,
            **kwargs
        )
        return handler(func)
    return decorator

def retry_on_http_error(max_retries: int = 3,
                       base_delay: float = 1.0,
                       **kwargs) -> Callable:
    """
    Decorator for retrying HTTP errors
    
    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries
        kwargs: Additional retry handler arguments
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        handler = HTTPRetryHandler(
            max_retries=max_retries,
            base_delay=base_delay,
            **kwargs
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return handler.retry(func, *args, **kwargs)
        
        return wrapper
    return decorator

# Common retry conditions
def is_temporary_error(exception: Exception) -> bool:
    """Check if exception represents a temporary error"""
    error_msg = str(exception).lower()
    temporary_indicators = [
        'timeout', 'connection', 'network', 'temporary', 
        'rate limit', 'throttle', 'busy', 'overload'
    ]
    return any(indicator in error_msg for indicator in temporary_indicators)

def is_rate_limit_error(exception: Exception) -> bool:
    """Check if exception represents a rate limit error"""
    error_msg = str(exception).lower()
    rate_limit_indicators = ['rate limit', 'throttle', '429', 'too many requests']
    return any(indicator in error_msg for indicator in rate_limit_indicators)