import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps

from aiosonic.exceptions import (
    BaseTimeout,
    ConnectionDisconnected,
    ConnectionPoolAcquireTimeout,
    ConnectTimeout,
    HttpParsingError,
    ReadTimeout,
    RequestTimeout,
)

DEFAULT_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    asyncio.TimeoutError,
    ConnectionError,
    BaseTimeout,
    ConnectTimeout,
    ReadTimeout,
    RequestTimeout,
    ConnectionPoolAcquireTimeout,
    ConnectionDisconnected,
    HttpParsingError,
)


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504)
    retry_on_exceptions: tuple[type[Exception], ...] = ()
    jitter: bool = True

    def __post_init__(self) -> None:
        if not self.retry_on_exceptions:
            # Provide sensible defaults including asyncio timeout and connection errors
            self.retry_on_exceptions = DEFAULT_RETRY_EXCEPTIONS


def retry_async(
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
):
    """Decorator for adding retry logic to async functions."""
    _config = config or RetryConfig()

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception: Exception | None = None

            for attempt in range(_config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    should_retry = isinstance(e, _config.retry_on_exceptions)

                    # For HTTP exceptions, check status code
                    if hasattr(e, "status_code") and hasattr(e, "args"):
                        if len(e.args) > 0 and isinstance(e.args[0], str):
                            import re

                            status_match = re.search(r"HTTP (\d+)", e.args[0])
                            if status_match:
                                status_code = int(status_match.group(1))
                                should_retry = status_code in _config.retry_on_status

                    if not should_retry or attempt == _config.max_attempts - 1:
                        raise

                    # Calculate delay with exponential backoff and jitter
                    delay = _config.base_delay * (_config.backoff_factor**attempt)
                    delay = min(delay, _config.max_delay)

                    if _config.jitter:
                        delay = delay * (0.5 + random.random() * 0.5)

                    if on_retry:
                        on_retry(attempt + 1, e)

                    await asyncio.sleep(delay)

            if last_exception:
                raise last_exception from None
            raise Exception("Retry failed without exception")

        return wrapper

    return decorator


def retry_http(
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
):
    """Decorator for adding HTTP-specific retry logic to async functions.

    Handles both HTTP status codes and network exceptions with proper retry logic.
    """
    _config = config or RetryConfig()

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception: Exception | None = None

            for attempt in range(_config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    should_retry = isinstance(e, _config.retry_on_exceptions)

                    # For HTTP exceptions, check status code using regex pattern
                    if not should_retry and hasattr(e, "args") and len(e.args) > 0:
                        import re

                        status_match = re.search(r"HTTP (\d+)", str(e.args[0]))
                        if status_match:
                            status_code = int(status_match.group(1))
                            should_retry = status_code in _config.retry_on_status

                    if not should_retry or attempt == _config.max_attempts - 1:
                        raise

                    # Calculate delay with exponential backoff and jitter
                    delay = _config.base_delay * (_config.backoff_factor**attempt)
                    delay = min(delay, _config.max_delay)

                    if _config.jitter:
                        delay = delay * (0.5 + random.random() * 0.5)

                    if on_retry:
                        on_retry(attempt + 1, e)

                    await asyncio.sleep(delay)

            if last_exception:
                raise last_exception from None
            raise Exception("Retry failed without exception")

        return wrapper

    return decorator
