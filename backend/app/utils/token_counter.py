"""
File: backend/app/utils/token_counter.py
Description: Fast token counting utility using tiktoken to enforce LLM context limits.
"""

import tiktoken
import logging

logger = logging.getLogger(__name__)

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Calculates the exact token count for a given text string.
    Defaults to 'cl100k_base' which is standard for most modern LLMs.
    """
    if not text:
        return 0
        
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"tiktoken encoding failed: {e}. Falling back to character heuristic.")
        # Fallback: ~4 characters per token
        return len(text) // 4