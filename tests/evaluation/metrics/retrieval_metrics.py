"""
File: tests/evaluation/metrics/retrieval_metrics.py
Description: Deterministic metric calculation functions for Hit@K, Recall@K, and MRR.
"""

from typing import Optional


def compute_first_relevant_rank(
    retrieved_keys: list[tuple[str, int]],
    expected_keys: set[tuple[str, int]],
) -> Optional[int]:
    """
    Finds the 1-based rank of the first retrieved chunk matching expected ground-truth keys.
    Returns None if no relevant chunk was retrieved.
    """
    for index, key in enumerate(retrieved_keys, start=1):
        if key in expected_keys:
            return index
    return None


def compute_hit_at_k(first_relevant_rank: Optional[int], k: int) -> bool:
    """Returns True if a relevant item was retrieved at or before rank K."""
    if first_relevant_rank is None:
        return False
    return first_relevant_rank <= k


def compute_recall_at_k(
    retrieved_keys: list[tuple[str, int]],
    expected_keys: set[tuple[str, int]],
    k: int,
) -> float:
    """
    Computes Recall@K: proportion of expected ground-truth chunks retrieved in top K.
    """
    if not expected_keys:
        return 0.0

    retrieved_in_k = set(retrieved_keys[:k])
    relevant_found = retrieved_in_k.intersection(expected_keys)
    return round(len(relevant_found) / len(expected_keys), 4)


def compute_reciprocal_rank(first_relevant_rank: Optional[int]) -> float:
    """Computes Reciprocal Rank (1/rank) for MRR calculation."""
    if first_relevant_rank is None or first_relevant_rank <= 0:
        return 0.0
    return round(1.0 / first_relevant_rank, 4)