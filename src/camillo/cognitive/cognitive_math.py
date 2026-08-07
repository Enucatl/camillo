from datetime import UTC, datetime
from math import exp, log


def _aware_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(UTC)


def calculate_activation(
    base_importance: float,
    access_count: int,
    last_accessed_at: datetime,
    *,
    decay_rate: float,
    now: datetime | None = None,
) -> float:
    """Calculate recall activation from importance, use count, and recency.

    Args:
        base_importance: Long-term usefulness score assigned during ingestion.
        access_count: Number of times the memory has been recalled.
        last_accessed_at: Timezone-aware timestamp from Postgres.
        decay_rate: Exponential decay rate per hour.
        now: Optional timezone-aware clock value for deterministic tests.

    Returns:
        A bounded activation score used to blend retrieval and memory strength.

    Raises:
        ValueError: If a caller supplies a naive datetime.
    """
    current_time = now or _aware_now()
    if current_time.tzinfo is None or last_accessed_at.tzinfo is None:
        raise ValueError("calculate_activation requires timezone-aware datetimes")
    elapsed = current_time - last_accessed_at
    hours_since_last_access = max(elapsed.total_seconds() / 3600, 0)
    decay_score = exp(-decay_rate * hours_since_last_access)
    activation = (base_importance * decay_score) + (log(access_count + 1) * 0.2)
    return max(0.0, min(activation, 1.5))
