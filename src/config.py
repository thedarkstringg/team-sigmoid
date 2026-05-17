from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # AI Provider
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
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