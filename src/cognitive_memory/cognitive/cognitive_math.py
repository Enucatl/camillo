from datetime import UTC, datetime
from math import exp, log


def _aware_now() -> datetime:
    return datetime.now(UTC)


def _coerce_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def calculate_activation(
    base_importance: float,
    access_count: int,
    last_accessed_at: datetime,
    *,
    decay_rate: float,
    now: datetime | None = None,
) -> float:
    current_time = now or _aware_now()
    elapsed = current_time - _coerce_aware(last_accessed_at)
    hours_since_last_access = max(elapsed.total_seconds() / 3600, 0)
    decay_score = exp(-decay_rate * hours_since_last_access)
    activation = (base_importance * decay_score) + (log(access_count + 1) * 0.2)
    return max(0.0, min(activation, 1.5))


def calculate_edge_decay(
    weight: float,
    last_co_accessed_at: datetime,
    *,
    decay_rate: float,
    now: datetime | None = None,
) -> float:
    current_time = now or _aware_now()
    elapsed = current_time - _coerce_aware(last_co_accessed_at)
    hours_since_last_access = max(elapsed.total_seconds() / 3600, 0)
    return max(0.0, weight * exp(-decay_rate * hours_since_last_access))
