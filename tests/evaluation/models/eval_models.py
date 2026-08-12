"""
File: tests/evaluation/models/eval_models.py
Description: Data models for evaluation dataset, query-level results, and aggregate reports.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


class ChunkKey(BaseModel):
    filename: str
    chunk_index: int

    def to_tuple(self) -> tuple[str, int]:
        return (self.filename, self.chunk_index)


class DatasetItem(BaseModel):
    id: int
    question: str
    reference_answer: Optional[str] = None
    ref_chunk_index: list[int]
    ref_chunk_id: Optional[list[str]] = None
    reference_document: str

    def get_expected_chunk_keys(self) -> set[tuple[str, int]]:
        """Returns set of (filename, chunk_index) tuples representing ground truth."""
        return {(self.reference_document, idx) for idx in self.ref_chunk_index}


class RetrievedChunkSummary(BaseModel):
    rank: int
    filename: str
    chunk_index: int
    score: float
    is_relevant: bool


class QuestionEvalResult(BaseModel):
    question_id: int
    question: str
    expected_chunks: list[dict[str, Any]]
    retrieved_chunks: list[RetrievedChunkSummary]
    first_relevant_rank: Optional[int] = None
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    hit_at_10: bool = False
    recall_at_k: float = 0.0
    reciprocal_rank: float = 0.0
    retrieval_latency_ms: float = 0.0
    error: Optional[str] = None


class AggregateEvalReport(BaseModel):
    timestamp: str
    target_endpoint: str
    configured_top_k: int
    total_questions: int
    successful_requests: int
    failed_requests: int
    hit_at_1_rate: float
    hit_at_3_rate: float
    hit_at_5_rate: float
    hit_at_10_rate: float
    mrr: float
    mean_recall_at_k: float
    average_latency_ms: float
    failed_retrieval_ids: list[int] = Field(
        default_factory=list,
        description="IDs of queries where expected chunk was not retrieved in top K.",
    )
    results: list[QuestionEvalResult]