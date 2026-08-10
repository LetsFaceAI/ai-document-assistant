import httpx
import os
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class NvidiaClient:
    def __init__(self):
        self.api_key = settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url

    async def get_chat_completion(self, model: str, messages: list) -> str:
        """
        Sends a chat completion request to the LLM API and safely extracts the text response.

        Args:
            model: The target model name or OpenRouter slug.
            messages: A list of role/content message dictionaries.

        Returns:
            The generated response string from the model.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"  # Capitalized 'Content-Type'
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.0
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(   
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=20.0  # Increased timeout slightly for longer query generations
                )

                # 1. Catch HTTP errors (like 400 Bad Request, 401 Unauthorized)
                if response.status_code != 200:
                    logger.error(f"LLM API HTTP {response.status_code} Error: {response.text}")
                    response.raise_for_status()

                data = response.json()

                # 2. Check if the API returned an error dictionary inside a 200 response
                if "error" in data:
                    logger.error(f"LLM API returned error payload: {data['error']}")
                    raise ValueError(f"LLM API Error: {data['error']}")

                # 3. Check if 'choices' key exists before indexing
                if "choices" not in data or not data["choices"]:
                    logger.error(f"LLM API response missing 'choices'. Full response: {data}")
                    raise KeyError(f"Response missing 'choices' array: {data}")

                # Safely return the generated message text
                return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            # Logs the exact message returned by OpenRouter/NVIDIA when status is 4xx/5xx
            logger.error(f"HTTP Status Error for model '{model}': {e.response.text}")
            raise RuntimeError(f"LLM API HTTP {e.response.status_code}: {e.response.text}") from e

        except Exception as e:
            logger.error(f"Unexpected error in get_chat_completion for model '{model}': {str(e)}")
            raise