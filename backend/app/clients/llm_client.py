import httpx
import os
from app.core.config import settings

class NvidiaClient:
    def __init__(self):
        self.api_key = settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url

    async def get_chat_completion(self, model: str, messages: list) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages":messages,
            "max_tokens": 1000,
            "temperature": 0.0
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(   
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10.0
            )

        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]