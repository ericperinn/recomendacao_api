from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL_DADOS: str = "postgresql+asyncpg://recomendacao:recomendacao@localhost:5432/recomendacao_dados"
    DATABASE_URL_EMBEDDINGS: str = "postgresql+asyncpg://recomendacao:recomendacao@localhost:5433/recomendacao_embeddings"
    BROKER_URL: str = "amqp://recomendacao:recomendacao@localhost:5672/"
    INDEX_QUEUE_NAME: str = "recomendacao.indexacao"
    MODEL_PATH: str = "paraphrase-multilingual-mpnet-base-v2"
    TOP_K_DEFAULT: int = 10
    ADMIN_TOKEN: str = "troque_em_producao"
    SCORE_WEIGHT_COSINE: float = 0.9
    SCORE_WEIGHT_OVERLAP: float = 0.1
    PCD_BONUS_EXCLUSIVE: float = 0.05
    MIN_SCORE_THRESHOLD: float = 0.02

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
