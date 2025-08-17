# core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_MODEL_NAME: str
    AZURE_OPENAI_EMBEDDING_MODEL: str

    class Config:
        env_file = ".env"

settings = Settings()