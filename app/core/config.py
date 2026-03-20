from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://recomendacao:recomendacao@localhost:5432/recomendacao"
    MODEL_PATH: str = "paraphrase-multilingual-mpnet-base-v2"
    TOP_K_DEFAULT: int = 10
    ADMIN_TOKEN: str = "troque_em_producao"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
