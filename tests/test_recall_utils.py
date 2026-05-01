import pytest

from camillo.cognitive.recall_utils import (
    Candidate,
    apply_diversity_filter,
    cosine_similarity,
    normalize_scores,
    reciprocal_rank_fusion,
)
from tests.fakes import make_memory


def test_rrf_merges_vector_and_text_results() -> None:
    """Protect the rank-fusion contract that lets two retrievers reinforce hits."""
    shared = make_memory("shared")
    vector_only = make_memory("vector only")
    text_only = make_memory("text only")

    candidates = reciprocal_rank_fusion(
        [(shared, 0.8), (vector_only, 0.7)],
        [(shared, 0.6), (text_only, 0.5)],
        rrf_k=60,
        limit=10,
    )

    assert candidates[0].memory.id == shared.id
    assert candidates[0].vector_score == 0.8
    assert candidates[0].text_score == 0.6
    assert candidates[0].rrf_score is not None
    assert candidates[0].rrf_score > (candidates[1].rrf_score or 0.0)


def test_normalize_scores_handles_equal_values() -> None:
    """Protect constant-score handling so useful all-positive signals survive."""
    assert normalize_scores([]) == []
    assert normalize_scores([2.0, 2.0]) == [1.0, 1.0]
    assert normalize_scores([0.0, 0.0]) == [0.0, 0.0]
    assert normalize_scores([2.0, 4.0]) == [0.0, 1.0]


def test_cosine_similarity() -> None:
    """Protect dependency-free vector comparisons used by diversity filtering."""
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 0]) == pytest.approx(0.0)


def test_diversity_filter_removes_near_duplicates() -> None:
    """Protect top-K variety so duplicate embeddings do not crowd the response."""
    first = Candidate(make_memory("first", embedding=[1.0, 0.0]), final_score=1.0)
    duplicate = Candidate(make_memory("duplicate", embedding=[0.99, 0.01]), final_score=0.9)
    distinct = Candidate(make_memory("distinct", embedding=[0.0, 1.0]), final_score=0.8)

    kept = apply_diversity_filter(
        [first, duplicate, distinct],
        similarity_threshold=0.92,
        max_results=3,
    )

    assert [candidate.memory.id for candidate in kept] == [first.memory.id, distinct.memory.id]
