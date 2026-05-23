import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # AI Provider
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = "https://openrouter.ai/api/v1"
    google_api_key: str = ""

    # Nutrition
    nutrition_provider: str = "usda"
    usda_api_key: str = ""

    # Storage
    database_url: str = "sqlite+aiosqlite:///./foodanalyzer.db"

    # Cache
    cache_ttl_seconds: int = 86400  # 24h

    # Concurrency
    nutrition_concurrency_limit: int = 10  # semaphore bound

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Logging
    log_level: str = "INFO"

    # Upload limits
    max_image_bytes: int = 5 * 1024 * 1024  # 5 MB


settings = Settings()

# Export to os.environ so ai/ providers pick them up via os.getenv
os.environ.setdefault("LLM_PROVIDER", settings.llm_provider)
os.environ.setdefault("LLM_MODEL", settings.llm_model)
os.environ.setdefault("OPENAI_BASE_URL", settings.openai_base_url)
os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
os.environ.setdefault("GOOGLE_API_KEY", settings.google_api_key)
os.environ.setdefault("USDA_API_KEY", settings.usda_api_key)