from datetime import UTC, datetime, timedelta

from cognitive_memory.cognitive.cognitive_math import calculate_activation, calculate_edge_decay


def test_activation_is_higher_for_recent_memory_than_stale_memory() -> None:
    now = datetime.now(UTC)

    recent = calculate_activation(0.8, 0, now - timedelta(minutes=5), decay_rate=0.1, now=now)
    stale = calculate_activation(0.8, 0, now - timedelta(days=7), decay_rate=0.1, now=now)

    assert recent > stale


def test_activation_increases_with_access_count() -> None:
    now = datetime.now(UTC)

    low_access = calculate_activation(0.5, 0, now, decay_rate=0.1, now=now)
    high_access = calculate_activation(0.5, 5, now, decay_rate=0.1, now=now)

    assert high_access > low_access


def test_edge_decay_lowers_old_edge_weight() -> None:
    now = datetime.now(UTC)

    fresh = calculate_edge_decay(2.0, now, decay_rate=0.1, now=now)
    old = calculate_edge_decay(2.0, now - timedelta(days=3), decay_rate=0.1, now=now)

    assert old < fresh


def test_activation_is_clamped() -> None:
    now = datetime.now(UTC)

    activation = calculate_activation(10.0, 1000, now, decay_rate=0.1, now=now)

    assert activation == 1.5
