from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:1234@localhost/test")

    class Config:
        env_file = ".env"

settings = Settings()