from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's input message")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The generated AI response")