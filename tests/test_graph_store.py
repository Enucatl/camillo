from uuid import uuid4

import pytest

from tests.fakes import FakeGraphStore


@pytest.mark.asyncio
async def test_reinforce_clique_deduplicates_ids_and_uses_pairs() -> None:
    """Protect reinforcement from duplicate IDs inflating edge weights."""
    graph_store = FakeGraphStore()
    first = uuid4()
    second = uuid4()
    third = uuid4()

    await graph_store.reinforce_clique([first, second, first, third], increment=2.0)

    assert len(graph_store.edges) == 3
    assert all(weight == 2.0 for weight in graph_store.edges.values())


@pytest.mark.asyncio
async def test_get_strong_neighbors_limits_and_excludes_sources() -> None:
    """Protect Hebbian spread from weak links and primary-result duplication."""
    graph_store = FakeGraphStore()
    source = uuid4()
    neighbor = uuid4()
    weak_neighbor = uuid4()
    other_source = uuid4()

    await graph_store.create_or_increment_edge(source, neighbor, increment=3.0)
    await graph_store.create_or_increment_edge(source, weak_neighbor, increment=1.0)
    await graph_store.create_or_increment_edge(source, other_source, increment=4.0)

    links = await graph_store.get_strong_neighbors(
        [source, other_source],
        min_weight=2.0,
        limit_per_source=1,
    )

    assert links == [(source, neighbor, 3.0)]
