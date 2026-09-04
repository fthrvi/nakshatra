import random
from typing import Optional


def delays(attempt: int, *, base: float = 0.5, cap: float = 60.0,
           jitter: float = 1.0, rand=None) -> float:
    """Return the seconds to wait before attempt number `attempt` (1-based)."""
    if attempt < 1:
        return 0.0

    # Clamp negative values to 0
    base = max(0.0, base)
    cap = max(0.0, cap)
    jitter = max(0.0, jitter)  # jitter can be any non-negative value

    # Exponential backoff: base * 2 ** (attempt - 1)
    exponential_delay = base * (2 ** (attempt - 1))

    # Clamp to cap
    clamped_delay = min(exponential_delay, cap)

    # Apply jitter: multiply by factor in [1 - jitter/2, 1 + jitter/2]
    if rand is None:
        rand = random.random

    factor_low = 1.0 - jitter / 2.0
    factor_high = 1.0 + jitter / 2.0
    factor = factor_low + (factor_high - factor_low) * rand()

    jittered_delay = clamped_delay * factor

    # Clamp to [0, cap]
    return max(0.0, min(jittered_delay, cap))


def should_retry(status: Optional[int], exc_name: str = "") -> bool:
    """Determine whether a failure is worth retrying."""
    # Retryable HTTP status codes
    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
    
    # Retryable exception names
    retryable_exceptions = {
        "TimeoutError", "ConnectionError", "ConnectionResetError",
        "URLError", "socket.timeout"
    }
    
    # Check status first
    if status is not None:
        if status in retryable_statuses:
            return True
        # Non-retryable statuses (explicitly not retrying)
        non_retryable_statuses = {400, 401, 403, 404, 409, 422}
        if status in non_retryable_statuses:
            return False
    
    # Check exception name
    if exc_name in retryable_exceptions:
        return True
    
    # If status=None and exc_name is empty or not matching, return False
    return False