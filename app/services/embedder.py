"""
Servico de embeddings com fallback leve para ambiente local.
Prioriza SentenceTransformer, alinhado ao notebook do Colab.
"""
import numpy as np

from app.core.config import settings


class EmbedderService:
    _instance: "EmbedderService | None" = None
    _model_path_loaded: str | None = None

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.backend = "hashing"
        self.model = None

        try:
            from sentence_transformers import SentenceTransformer

            print(f"[Embedder] Carregando SentenceTransformer: {model_path}")
            self.model = SentenceTransformer(model_path)
            self.backend = "sentence-transformers"
            print("[Embedder] Backend SBERT carregado.")
        except Exception as exc:
            print(f"[Embedder] Falha ao carregar SBERT ({model_path}): {exc}")
            print("[Embedder] Usando fallback hashing.")
            from sklearn.feature_extraction.text import HashingVectorizer

            self.vectorizer = HashingVectorizer(
                n_features=768,
                alternate_sign=False,
                norm=None,
                ngram_range=(1, 2),
            )
            print("[Embedder] Backend fallback carregado.")

    def encode(self, textos: list[str]) -> np.ndarray:
        """Retorna matriz (N, 768) com vetores L2-normalizados."""
        if not textos:
            return np.empty((0, 768), dtype=np.float32)

        if self.backend == "sentence-transformers" and self.model is not None:
            return self.model.encode(
                textos,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)

        sparse = self.vectorizer.transform(textos)
        dense = sparse.toarray().astype(np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms

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
