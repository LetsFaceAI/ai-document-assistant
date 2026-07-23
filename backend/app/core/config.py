from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Define fields with their expected types and default values
    app_name: str = "AI Document Assistant"
    port: int = 8000
    debug_mode: bool = False
    openrouter_api_key: str = "FakeKey123"

    # Tell pydantic where to find the .env files
    model_config = SettingsConfigDict(env_file=".env", 
    env_file_encoding="utf-8", extra="ignore")


settings = Settings()