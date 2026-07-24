from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(...,description="The Status of the Health Check")
    app_name: str = Field(...,description="The Name of the Application")
    environment: str = Field(...,description="The Environment of the Application")