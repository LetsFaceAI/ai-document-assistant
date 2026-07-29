import logging
import re
import time
import unicodedata
from typing import Callable, Dict, List

from app.models.cleaning import TextCleaningResult

logger = logging.getLogger(__name__)


class TextCleaningService:
    """
    A modular text cleaning pipeline designed to prepare raw PDF text for RAG embedding.
    Preserves semantic meaning and paragraph boundaries while fixing extraction artifacts.
    """

    def __init__(self, active_rules: List[str] | None = None):
        # 1. Define all available modular rules mapping
        self._available_rules: Dict[str, Callable[[str], str]] = {
            "normalize_unicode": self._normalize_unicode,
            "standardize_line_endings": self._standardize_line_endings,
            "remove_control_characters": self._remove_control_characters,
            "standardize_quotes_and_dashes": self._standardize_quotes_and_dashes,
            "repair_broken_urls": self._repair_broken_urls,
            "repair_hyphenation": self._repair_hyphenation,
            "repair_mid_sentence_breaks": self._repair_mid_sentence_breaks,
            "remove_page_numbers": self._remove_page_numbers,
            "remove_noise_and_dividers": self._remove_noise_and_dividers,
            "normalize_whitespace": self._normalize_whitespace,
            "preserve_paragraphs": self._preserve_paragraphs,
        }

        # 2. Define the execution order. If none provided, use all default rules.
        self.active_rules = active_rules or list(self._available_rules.keys())

        # Validate provided rules
        for rule in self.active_rules:
            if rule not in self._available_rules:
                raise ValueError(f"Unknown cleaning rule: '{rule}'")

    def clean_text(self, raw_text: str) -> TextCleaningResult:
        """Runs the raw text through the configured active rule pipeline."""
        if not raw_text:
            return TextCleaningResult(
                original_text="",
                cleaned_text="",
                original_char_count=0,
                cleaned_char_count=0,
                applied_rules_count=0,
                processing_time_ms=0.0
            )

        start_time = time.perf_counter()
        current_text = raw_text

        # Execute active rules sequentially
        for rule_name in self.active_rules:
            rule_func = self._available_rules[rule_name]
            current_text = rule_func(current_text)

        end_time = time.perf_counter()
        processing_time_ms = round((end_time - start_time) * 1000, 2)

        logger.info(
            f"Cleaned Text: {len(raw_text):,} -> {len(current_text):,} chars | "
            f"Rules: {len(self.active_rules)} | Time: {processing_time_ms}ms"
        )

        return TextCleaningResult(
            original_text=raw_text,
            cleaned_text=current_text,
            original_char_count=len(raw_text),
            cleaned_char_count=len(current_text),
            applied_rules_count=len(self.active_rules),
            processing_time_ms=processing_time_ms
        )

    # ==========================================
    # MODULAR CLEANING RULES (Isolated Logic)
    # ==========================================

    def _normalize_unicode(self, text: str) -> str:
        """Normalizes Unicode characters (e.g., ligatures like 'fi' -> 'fi')."""
        return unicodedata.normalize("NFKC", text)

    def _standardize_line_endings(self, text: str) -> str:
        """Converts Windows/Mac line endings to Unix standard."""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _remove_control_characters(self, text: str) -> str:
        """Strips non-printable control characters but keeps tabs and newlines."""
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    def _standardize_quotes_and_dashes(self, text: str) -> str:
        """Standardizes curly quotes to straight quotes, and various dashes to a simple hyphen."""
        text = re.sub(r"[“”]", '"', text)
        text = re.sub(r"[‘’]", "'", text)
        return re.sub(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]", "-", text)

    def _repair_broken_urls(self, text: str) -> str:
        """Rejoins URLs that were split across multiple lines."""
        return re.sub(r"(https?://[a-zA-Z0-9./-]+)\s*\n\s*([a-zA-Z0-9./-]+)", r"\1\2", text)

    def _repair_hyphenation(self, text: str) -> str:
        """Rejoins words split by a hyphen at the end of a line (e.g., 'sys-\n tem' -> 'system')."""
        return re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    def _repair_mid_sentence_breaks(self, text: str) -> str:
        """
        Removes single newlines that interrupt sentences. 
        Looks for a lowercase letter or punctuation, a newline, and another lowercase letter.
        """
        return re.sub(r"(?<=[a-z,;:])\s*\n\s*(?=[a-z])", " ", text)

    def _remove_page_numbers(self, text: str) -> str:
        """Strips standalone lines containing only page numbers (e.g., 'Page 12' or '- 12 -')."""
        return re.sub(r"(?m)^\s*(?:page|pg\.?)?\s*-?\s*\d+\s*-?\s*$\n?", "", text, flags=re.IGNORECASE)

    def _remove_noise_and_dividers(self, text: str) -> str:
        """Filters out lines containing only repeating symbols (e.g., '-------') or tiny isolated artifacts."""
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Drop lines consisting purely of 4 or more repeating non-alphanumeric characters
            if re.match(r"^([^a-zA-Z0-9])\1{3,}$", stripped):
                continue
            # Drop tiny 1-2 character lines that contain no letters or numbers (e.g., stray bullets)
            if len(stripped) > 0 and len(stripped) < 3 and not re.search(r"[a-zA-Z0-9]", stripped):
                continue
            cleaned_lines.append(stripped)
            
        return "\n".join(cleaned_lines)

    def _normalize_whitespace(self, text: str) -> str:
        """Converts tabs/non-breaking spaces to standard spaces and collapses multiple horizontal spaces."""
        text = text.replace("\xa0", " ").replace("\t", " ")
        # Collapse 2+ horizontal spaces into 1, ignoring newlines
        return re.sub(r"[ ]{2,}", " ", text)

    def _preserve_paragraphs(self, text: str) -> str:
        """
        Ensures paragraphs are separated by exactly two newlines. 
        Collapses 3+ newlines into 2, preventing the document from flattening into one paragraph.
        """
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()