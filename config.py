from typing import Literal

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_model: str
    ollama_host: AnyHttpUrl
    refund_approval_threshold_usd: int = Field(ge=0)
    log_level: Literal["debug", "info", "warning", "error"] = "debug"


settings = Settings()
