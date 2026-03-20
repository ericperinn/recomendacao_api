"""
Serviço de embeddings — singleton com modelo trocável via config.
Para trocar o modelo:
  1. Atualize MODEL_PATH no .env
  2. Chame POST /admin/recalcular-embeddings
  3. A instância é regerada automaticamente no próximo request
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings


class EmbedderService:
    _instance: "EmbedderService | None" = None
    _model_path_loaded: str | None = None

    def __init__(self, model_path: str):
        print(f"[Embedder] Carregando modelo: {model_path}")
        self.model = SentenceTransformer(model_path)
        self.model.max_seq_length = 512
        self.model_path = model_path
        print(f"[Embedder] Modelo carregado.")

    def encode(self, textos: list[str]) -> np.ndarray:
        """Retorna matriz (N, dim) com embeddings L2-normalizados."""
        return self.model.encode(
            textos,
            show_progress_bar=False,
            normalize_embeddings=True,  # facilita usar produto escalar como similaridade
            batch_size=32,
        )

    @classmethod
    def get(cls) -> "EmbedderService":
        """
        Retorna o singleton.
        Se MODEL_PATH mudou desde a última carga, recria a instância.
        """
        current_path = settings.MODEL_PATH
        if cls._instance is None or cls._model_path_loaded != current_path:
            cls._instance = cls(current_path)
            cls._model_path_loaded = current_path
        return cls._instance

    @classmethod
    def reload(cls) -> "EmbedderService":
        """Força recarga do modelo (usado pelo endpoint de admin)."""
        cls._instance = None
        return cls.get()
