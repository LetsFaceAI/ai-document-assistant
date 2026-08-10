"""
File: backend/app/services/query_expansion_service.py
Description: Robust query expansion service configured for OpenRouter / Nvidia models.
"""

import json
import logging
import re
from typing import List
from app.clients.llm_client import NvidiaClient

logger = logging.getLogger(__name__)

class QueryExpansionService:
    def __init__(self, llm_client: NvidiaClient, model_name: str, query_count: int = 4):
        self.client = llm_client
        self.model_name = model_name
        self.query_count = query_count

    async def generate_expanded_queries(self, original_query: str) -> List[str]:
        """
        Generates alternative search queries and parses OpenRouter output cleanly.
        """
        prompt = f"""
        Generate {self.query_count - 1} search queries that explore different ways someone might express the following information need.
        Avoid changing the meaning. Each query should emphasize different terminology or phrasing.
        Do not answer the question. Only generate search queries.
        
        Return strictly a JSON array of strings like this:
        ["query variation 1", "query variation 2", "query variation 3"]
        
        Original Query: "{original_query}"
        """

        messages = [
            {"role": "system", "content": "You are a search query optimization assistant. You output strictly JSON arrays of strings."},
            {"role": "user", "content": prompt}
        ]

        try:
            raw_response = await self.client.get_chat_completion(
                model=self.model_name, 
                messages=messages
            )

            # Strip whitespace and markdown backticks if present
            clean_text = raw_response.strip()
            if "```" in clean_text:
                # Find content inside code blocks
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean_text, re.DOTALL)
                if match:
                    clean_text = match.group(1).strip()

            # Attempt JSON parsing
            try:
                generated_queries = json.loads(clean_text)
                if isinstance(generated_queries, dict):
                    generated_queries = list(generated_queries.values())[0]
            except json.JSONDecodeError:
                # Fallback: Parse line-by-line if model outputs a numbered list
                lines = [line.strip() for line in clean_text.split("\n") if line.strip()]
                generated_queries = [re.sub(r"^\d+[\.\)\-]\s*", "", line) for line in lines]

        except Exception as e:
            logger.error(f"LLM query generation failed: {e}")
            # Graceful fallback: Proceed using only the original user query
            return [original_query]

        # --- Deduplication Pipeline ---
        final_queries = [original_query]
        seen_queries = {original_query.strip().lower()}

        if isinstance(generated_queries, list):
            for q in generated_queries:
                if not isinstance(q, str):
                    continue
                clean_q = " ".join(q.split()).lower()
                if clean_q not in seen_queries and clean_q != "":
                    seen_queries.add(clean_q)
                    final_queries.append(q.strip())

        logger.info(f"Generated {len(final_queries)} unique search queries.")
        return final_queries