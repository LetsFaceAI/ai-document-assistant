import re
import math
import uuid
import logging
from typing import List, Tuple

from app.core.config import settings
from app.models.chunk import DocumentChunk, ChunkingSummary

logger = logging.getLogger(__name__)

class ChunkingService:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.min_chunk_size = min_chunk_size or settings.MIN_CHUNK_SIZE

        # Hierarchical separators
        self.separators = [
            r"\n\n+",            # Rule 1: Paragraphs
            r"(?<=[.!?])\s+",     # Rule 2: Sentences
            r"\s+",              # Rule 3: Words
            ""                   # Rule 4: Characters
        ]

    def _estimate_tokens(self, text: str) -> int:
        """Approximates token count (~4 chars per token)."""
        return math.ceil(len(text) / 4) if text else 0

    def _split_recursively(self, text: str, sep_idx: int = 0) -> List[str]:
        """Recursively breaks text down into pieces <= chunk_size."""
        if len(text) <= self.chunk_size or sep_idx >= len(self.separators):
            return [text] if text else []

        sep = self.separators[sep_idx]
        if sep == "":
            # Rule 4: Character fallback
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        splits = re.split(sep, text)
        pieces = []
        for s in splits:
            s_stripped = s.strip()
            if not s_stripped:
                continue
            if len(s_stripped) <= self.chunk_size:
                pieces.append(s_stripped)
            else:
                # Drill down to next finest separator
                pieces.extend(self._split_recursively(s_stripped, sep_idx + 1))
        return pieces

    def create_chunks(self, text: str) -> Tuple[List[DocumentChunk], ChunkingSummary]:
        if not text or not text.strip():
            logger.warning("Empty text passed to ChunkingService")
            empty_summary = ChunkingSummary(
                total_chunks=0,
                avg_chunk_size=0.0,
                largest_chunk_size=0,
                smallest_chunk_size=0,
                total_estimated_tokens=0,
            )
            return [], empty_summary

        # 1. Break text into atomic semantic elements under chunk_size
        pieces = self._split_recursively(text)

        # 2. Re-combine pieces with sliding overlap window
        raw_chunks: List[str] = []
        current_chunk = ""

        for piece in pieces:
            if current_chunk and (len(current_chunk) + 1 + len(piece) > self.chunk_size):
                raw_chunks.append(current_chunk.strip())

                # Apply chunk overlap window
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                overlap_text = current_chunk[overlap_start:]
                current_chunk = (overlap_text + " " + piece).strip()
            else:
                current_chunk = (current_chunk + " " + piece).strip() if current_chunk else piece

        if current_chunk:
            raw_chunks.append(current_chunk.strip())

        # 3. Create DocumentChunk models and compute text offsets
        final_chunks: List[DocumentChunk] = []
        last_end = 0

        for chunk_text in raw_chunks:
            # Filter out chunks smaller than min_chunk_size unless it's the only one
            if len(chunk_text) < self.min_chunk_size and len(raw_chunks) > 1:
                continue

            # Determine position in original text
            start_offset = text.find(chunk_text[:30], max(0, last_end - self.chunk_overlap - 50))
            if start_offset == -1:
                start_offset = text.find(chunk_text[:20])
                if start_offset == -1:
                    start_offset = last_end

            end_offset = start_offset + len(chunk_text)
            last_end = end_offset

            chunk_obj = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                chunk_index=len(final_chunks),
                text=chunk_text,
                character_count=len(chunk_text),
                estimated_token_count=self._estimate_tokens(chunk_text),
                start_offset=start_offset,
                end_offset=end_offset,
            )
            final_chunks.append(chunk_obj)

        # 4. Generate Summary Statistics
        if not final_chunks:
            empty_summary = ChunkingSummary(
                total_chunks=0,
                avg_chunk_size=0.0,
                largest_chunk_size=0,
                smallest_chunk_size=0,
                total_estimated_tokens=0,
            )
            return [], empty_summary

        sizes = [c.character_count for c in final_chunks]
        total_tokens = sum(c.estimated_token_count for c in final_chunks)

        summary = ChunkingSummary(
            total_chunks=len(final_chunks),
            avg_chunk_size=round(sum(sizes) / len(final_chunks), 2),
            largest_chunk_size=max(sizes),
            smallest_chunk_size=min(sizes),
            total_estimated_tokens=total_tokens,
        )

        logger.info(
            f"Chunking complete: {summary.total_chunks} chunks created "
            f"(Avg size: {summary.avg_chunk_size} chars, Total est. tokens: {summary.total_estimated_tokens})"
        )

        return final_chunks, summary