"""
File: tests/evaluation/runners/retrieval_evaluator.py
Description: Asynchronous evaluation runner for deterministic RAG retrieval benchmarking.
Usage:
    python tests/evaluation/runners/retrieval_evaluator.py --base-url http://localhost:8000/api/v1 --top-k 10
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
import httpx
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from tests.evaluation.models.eval_models import (
    DatasetItem,
    QuestionEvalResult,
    RetrievedChunkSummary,
    AggregateEvalReport,
)
from tests.evaluation.metrics.retrieval_metrics import (
    compute_first_relevant_rank,
    compute_hit_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RetrievalEvaluator")


class RetrievalEvaluator:
    def __init__(self, base_url: str, top_k: int, dataset_path: Path, output_dir: Path):
        self.base_url = base_url.rstrip("/")
        self.search_url = f"{self.base_url}/search"
        self.top_k = top_k
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self) -> list[DatasetItem]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        return [DatasetItem(**item) for item in raw_data]

    async def evaluate_single_question(
        self, client: httpx.AsyncClient, item: DatasetItem
    ) -> QuestionEvalResult:
        expected_keys = item.get_expected_chunk_keys()
        payload = {"question": item.question, "top_k": self.top_k}

        try:
            response = await client.post(self.search_url, json=payload, timeout=60.0)
            if response.status_code != 200:
                logger.error(f"[Q ID {item.id}] HTTP Error {response.status_code}: {response.text}")
                return QuestionEvalResult(
                    question_id=item.id,
                    question=item.question,
                    expected_chunks=[{"filename": item.reference_document, "chunk_index": idx} for idx in item.ref_chunk_index],
                    retrieved_chunks=[],
                    error=f"HTTP {response.status_code}: {response.text}",
                )

            data = response.json()
            retrieved_chunks_data = data.get("retrieved_chunks", [])
            latency_ms = float(data.get("retrieval_time_ms", 0.0))

            retrieved_summaries: list[RetrievedChunkSummary] = []
            retrieved_keys: list[tuple[str, int]] = []

            for rank, chunk in enumerate(retrieved_chunks_data, start=1):
                fname = chunk.get("filename", "")
                c_idx = int(chunk.get("chunk_index", -1))
                score = float(chunk.get("score", 0.0))
                key = (fname, c_idx)

                is_rel = key in expected_keys
                retrieved_keys.append(key)
                retrieved_summaries.append(
                    RetrievedChunkSummary(
                        rank=rank,
                        filename=fname,
                        chunk_index=c_idx,
                        score=score,
                        is_relevant=is_rel,
                    )
                )

            first_rank = compute_first_relevant_rank(retrieved_keys, expected_keys)
            reciprocal_rank = compute_reciprocal_rank(first_rank)
            recall = compute_recall_at_k(retrieved_keys, expected_keys, self.top_k)

            rank_str = f"Rank #{first_rank}" if first_rank else "NOT FOUND"
            logger.info(
                f"[Q ID {item.id:02d}] {item.question[:45]}... -> {rank_str} ({latency_ms:.1f}ms)"
            )

            return QuestionEvalResult(
                question_id=item.id,
                question=item.question,
                expected_chunks=[{"filename": item.reference_document, "chunk_index": idx} for idx in item.ref_chunk_index],
                retrieved_chunks=retrieved_summaries,
                first_relevant_rank=first_rank,
                hit_at_1=compute_hit_at_k(first_rank, 1),
                hit_at_3=compute_hit_at_k(first_rank, 3),
                hit_at_5=compute_hit_at_k(first_rank, 5),
                hit_at_10=compute_hit_at_k(first_rank, 10),
                recall_at_k=recall,
                reciprocal_rank=reciprocal_rank,
                retrieval_latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"[Q ID {item.id}] Exception during request: {str(e)}")
            return QuestionEvalResult(
                question_id=item.id,
                question=item.question,
                expected_chunks=[{"filename": item.reference_document, "chunk_index": idx} for idx in item.ref_chunk_index],
                retrieved_chunks=[],
                error=str(e),
            )

    async def run_evaluation(self) -> AggregateEvalReport:
        dataset = self.load_dataset()
        logger.info(f"Loaded {len(dataset)} items from dataset. Endpoint: {self.search_url}")

        results: list[QuestionEvalResult] = []
        async with httpx.AsyncClient() as client:
            for idx, item in enumerate(dataset, start=1):
                logger.info(f"Progress: [{idx}/{len(dataset)}]")
                eval_res = await self.evaluate_single_question(client, item)
                results.append(eval_res)

        # Compute Aggregates
        total = len(results)
        successful = [r for r in results if r.error is None]
        succ_count = len(successful)
        fail_req_count = total - succ_count

        hit1_count = sum(1 for r in successful if r.hit_at_1)
        hit3_count = sum(1 for r in successful if r.hit_at_3)
        hit5_count = sum(1 for r in successful if r.hit_at_5)
        hit10_count = sum(1 for r in successful if r.hit_at_10)

        mrr_sum = sum(r.reciprocal_rank for r in successful)
        recall_sum = sum(r.recall_at_k for r in successful)
        latency_sum = sum(r.retrieval_latency_ms for r in successful)

        failed_retrieval_ids = [
            r.question_id for r in successful if r.first_relevant_rank is None or r.first_relevant_rank > self.top_k
        ]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = AggregateEvalReport(
            timestamp=timestamp,
            target_endpoint=self.search_url,
            configured_top_k=self.top_k,
            total_questions=total,
            successful_requests=succ_count,
            failed_requests=fail_req_count,
            hit_at_1_rate=round(hit1_count / succ_count, 4) if succ_count else 0.0,
            hit_at_3_rate=round(hit3_count / succ_count, 4) if succ_count else 0.0,
            hit_at_5_rate=round(hit5_count / succ_count, 4) if succ_count else 0.0,
            hit_at_10_rate=round(hit10_count / succ_count, 4) if succ_count else 0.0,
            mrr=round(mrr_sum / succ_count, 4) if succ_count else 0.0,
            mean_recall_at_k=round(recall_sum / succ_count, 4) if succ_count else 0.0,
            average_latency_ms=round(latency_sum / succ_count, 2) if succ_count else 0.0,
            failed_retrieval_ids=failed_retrieval_ids,
            results=results,
        )

        self._save_reports(report)
        return report

    def _save_reports(self, report: AggregateEvalReport):
        # 1. JSON Report
        json_path = self.output_dir / f"retrieval_eval_{report.timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        # 2. Markdown Summary Report
        md_path = self.output_dir / f"retrieval_eval_{report.timestamp}.md"
        md_content = self._generate_markdown_summary(report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Reports saved to:\n - {json_path}\n - {md_path}")

    def _generate_markdown_summary(self, report: AggregateEvalReport) -> str:
        md = [
            f"# RAG Deterministic Retrieval Evaluation Report",
            f"",
            f"- **Timestamp:** `{report.timestamp}`",
            f"- **Target Endpoint:** `{report.target_endpoint}`",
            f"- **Configured Top-K:** `{report.configured_top_k}`",
            f"- **Total Questions Evaluated:** `{report.total_questions}`",
            f"- **Successful Requests:** `{report.successful_requests}`",
            f"- **Failed Requests:** `{report.failed_requests}`",
            f"",
            f"## Aggregate Metrics",
            f"",
            f"| Metric | Score | Percentage / Value |",
            f"| :--- | :--- | :--- |",
            f"| **Hit@1** | {report.hit_at_1_rate:.4f} | {report.hit_at_1_rate * 100:.2f}% |",
            f"| **Hit@3** | {report.hit_at_3_rate:.4f} | {report.hit_at_3_rate * 100:.2f}% |",
            f"| **Hit@5** | {report.hit_at_5_rate:.4f} | {report.hit_at_5_rate * 100:.2f}% |",
            f"| **Hit@10** | {report.hit_at_10_rate:.4f} | {report.hit_at_10_rate * 100:.2f}% |",
            f"| **MRR (Mean Reciprocal Rank)** | **{report.mrr:.4f}** | — |",
            f"| **Mean Recall@{report.configured_top_k}** | {report.mean_recall_at_k:.4f} | {report.mean_recall_at_k * 100:.2f}% |",
            f"| **Average Latency** | — | **{report.average_latency_ms:.2f} ms** |",
            f"",
            f"## Failed Retrieval Queries (Not Found in Top {report.configured_top_k})",
            f"",
        ]

        if not report.failed_retrieval_ids:
            md.append(" *All expected reference chunks were successfully retrieved!*")
        else:
            md.append("| Question ID | Question | Expected (Document, Chunk Index) |")
            md.append("| :--- | :--- | :--- |")
            for res in report.results:
                if res.question_id in report.failed_retrieval_ids:
                    expected_str = ", ".join([f"`({e['filename']}, {e['chunk_index']})`" for e in res.expected_chunks])
                    md.append(f"| {res.question_id} | {res.question} | {expected_str} |")

        return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="RAG Deterministic Retrieval Evaluator")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/api/v1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("tests/evaluation/dataset/rag_evaluation.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tests/evaluation/reports")
    )

    args = parser.parse_args()

    evaluator = RetrievalEvaluator(
        base_url=args.base_url,
        top_k=args.top_k,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
    )

    asyncio.run(evaluator.run_evaluation())


if __name__ == "__main__":
    main()