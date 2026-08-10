from app.clients.llm_client import NvidiaClient
from app.core.config import settings

class ChatService:
    def __init__(self):
        self.client = NvidiaClient()

    async def process_user_message(self, user_message: str) -> str:
        messages = [
            {"role": "system", "content": "you are smart information bot"},
            {"role": "user", "content": user_message}
        ]
        
        # Business logic: Model selection
        model_name = settings.openrouter_base_model

        # Delegate network execution to the client
        return await self.client.get_chat_completion(model=model_name, messages=messages)