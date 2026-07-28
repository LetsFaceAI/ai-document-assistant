import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


class TextCleaningService:
    """Service dedicated to normalizing and cleaning raw extracted PDF text for downstream NLP/RAG tasks."""

    def clean_text(self, raw_text: str) -> str:
        """
        Runs raw text through a multi-stage cleaning pipeline:
        1. Unicode Normalization (NFKC)
        2. Line ending standardization
        3. Control character removal
        4. Tab & Non-breaking space normalization
        5. Hyphenated word repair across line breaks
        6. Mid-sentence artificial newline joining
        7. Page number / standalone digit line stripping
        8. Line-by-line trimming & space collapsing
        9. Blank line normalization
        """
        if not raw_text:
            logger.warning("Empty raw text provided for cleaning.")
            return ""

        original_char_count = len(raw_text)
        logger.info(f"Cleaning Started | Original Characters: {original_char_count:,}")

        # 1. Unicode Normalization (NFKC replaces ligatures like 'fi' -> 'fi', smart quotes, accented glyphs)
        text = unicodedata.normalize("NFKC", raw_text)

        # 2. Normalize Line Endings (CRLF / CR -> Unix LF)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Strip Non-Printable Control Characters (Retains \n [0x0A] and \t [0x09])
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # 4. Convert Non-Breaking Spaces and Tabs to Standard Spaces
        text = text.replace("\xa0", " ").replace("\t", " ")

        # 5. Fix Hyphenated Words Split Across Lines (e.g., 'environ-\n ment' -> 'environment')
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

        # 6. Join Mid-Sentence Artificial Line Breaks (lowercase/comma followed by newline & lowercase)
        text = re.sub(r"(?<=[a-z,])\s*\n\s*(?=[a-z])", " ", text)

        # 7. Remove Standalone Page Numbers and Boilerplate Lines (e.g., 'Page 12', '- 12 -', '12')
        text = re.sub(r"(?m)^\s*(?:page|pg\.?)?\s*-?\s*\d+\s*-?\s*$\n?", "", text, flags=re.IGNORECASE)

        # 8. Strip Leading and Trailing Whitespace from Every Line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # 9. Collapse Multiple Consecutive Horizontal Spaces
        text = re.sub(r"[ \t]{2,}", " ", text)

        # 10. Collapse Excessive Empty Lines (Allows max 2 newlines = 1 blank line between paragraphs)
        text = re.sub(r"\n{3,}", "\n\n", text)

        cleaned_text = text.strip()
        cleaned_char_count = len(cleaned_text)

        logger.info(
            f"Cleaning Completed | Original Characters: {original_char_count:,}, "
            f"Cleaned Characters: {cleaned_char_count:,}"
        )

        return cleaned_text