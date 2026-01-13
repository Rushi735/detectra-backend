from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Detectra API"
    # Production: Use DATABASE_URL from Render/Neon
    # Development: Fallback to SQLite
    SQLALCHEMY_DATABASE_URI: str = os.getenv("DATABASE_URL", "sqlite:///./detectra.db")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key-change-this-in-prod")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
