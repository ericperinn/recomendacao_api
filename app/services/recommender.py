"""
Lógica central de recomendação.
Busca embeddings pré-calculados do banco e computa similaridade do cosseno.
Como os embeddings já estão normalizados (L2), produto escalar == cosine similarity.
"""
import re
import unicodedata

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema import CandidatoEmbedding, VagaEmbedding
from app.core.config import settings
_OVERLAP_STOP_TOKENS = {
    "pcd",
    "vaga",
    "vagas",
    "oportunidade",
    "exclusiva",
    "exclusivo",
    "preferencia",
    "preferencial",
}

EMBEDDING_DIM = 768  # paraphrase-multilingual-mpnet-base-v2


def _deserialize(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _tokenize_for_overlap(text: str | None) -> set[str]:
    if not text:
        return set()

    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    tokens = set(re.findall(r"[a-z0-9]+", normalized.lower()))
    return {token for token in tokens if token not in _OVERLAP_STOP_TOKENS}


def _overlap_ratio(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0

    union = left_tokens | right_tokens
    if not union:
        return 0.0

    return len(left_tokens & right_tokens) / len(union)


def _combined_score(cosine, overlap):
    return (
        np.asarray(cosine, dtype=np.float32) * float(settings.SCORE_WEIGHT_COSINE)
        + np.asarray(overlap, dtype=np.float32) * float(settings.SCORE_WEIGHT_OVERLAP)
    )


async def _load_embeddings_vagas(
    db: AsyncSession,
    filtrar_pcd: bool | None = None,
) -> list[VagaEmbedding]:
    """Carrega todos os embeddings de vagas, com filtro PCD opcional."""
    stmt = select(VagaEmbedding)
    if filtrar_pcd is True:
        stmt = stmt.where(VagaEmbedding.is_pcd_exclusive == True)
    elif filtrar_pcd is False:
        # vagas de ampla concorrência apenas
        stmt = stmt.where(VagaEmbedding.is_pcd_exclusive == False)
    result = await db.execute(stmt)
    return result.scalars().all()


async def _load_embeddings_candidatos(
    db: AsyncSession,
    apenas_pcd: bool = False,
) -> list[CandidatoEmbedding]:
    stmt = select(CandidatoEmbedding)
    if apenas_pcd:
        stmt = stmt.where(CandidatoEmbedding.is_pcd == True)
    result = await db.execute(stmt)
    return result.scalars().all()


# ── Candidato → Vagas ────────────────────────────────────────────────────────

async def recomendar_vagas(
    cand_id: int,
    top_k: int,
    db: AsyncSession,
    debug: bool = False,
) -> dict:
    """
    Dado um candidato, retorna as top_k vagas mais similares.
    Regras PCD:
      - Candidato PCD: recebe bônus (+0.25) em vagas exclusivas PCD
      - Candidato não-PCD: vagas exclusivas PCD são zeradas
    """
    # Busca embedding do candidato
    cand_row = await db.get(CandidatoEmbedding, cand_id)
    if cand_row is None:
        return {"error": f"Candidato {cand_id} não encontrado ou sem embedding gerado."}

    emb_cand = _deserialize(cand_row.embedding)
    candidato_pcd = cand_row.is_pcd
    cand_tokens = _tokenize_for_overlap(cand_row.texto_limpo)

    # Busca todos embeddings de vagas
    vagas = await _load_embeddings_vagas(db)
    if not vagas:
        return {
            "cand_id": cand_id,
            "is_pcd": bool(candidato_pcd),
            "total_retornado": 0,
            "vagas": [],
        }

    emb_vagas = np.stack([_deserialize(v.embedding) for v in vagas])
    cosine_scores = emb_vagas @ emb_cand

    overlap_scores = np.array(
        [
            _overlap_ratio(
                cand_tokens,
                _tokenize_for_overlap(
                    " ".join(
                        part for part in (vaga.vaga_titulo, vaga.vaga_area, vaga.texto_limpo) if part
                    )
                ),
            )
            for vaga in vagas
        ],
        dtype=np.float32,
    )

    scores = _combined_score(cosine_scores, overlap_scores)

    # Regras PCD
    for i, vaga in enumerate(vagas):
        if vaga.is_pcd_exclusive:
            if candidato_pcd:
                scores[i] += float(settings.PCD_BONUS_EXCLUSIVE)
            else:
                scores[i] = 0.0     # zera: candidato não-PCD não vê vaga exclusiva PCD

    # Ranqueia
    top_indices = np.argsort(scores)[::-1]
    top_indices = [i for i in top_indices if scores[i] >= float(settings.MIN_SCORE_THRESHOLD)][:top_k]

    response = {
        "cand_id": cand_id,
        "is_pcd": bool(candidato_pcd),
        "total_retornado": len(top_indices),
        "vagas": [
            {
                "vaga_id":        vagas[i].vaga_id,
                "titulo":         vagas[i].vaga_titulo,
                "area":           vagas[i].vaga_area,
                "is_pcd_exclusive": bool(vagas[i].is_pcd_exclusive),
                "score":          round(float(scores[i]), 4),
            }
            for i in top_indices
        ],
    }

    if debug:
        response["debug"] = [
            {
                "vaga_id": vagas[i].vaga_id,
                "titulo": vagas[i].vaga_titulo,
                "cosine": round(float(cosine_scores[i]), 6),
                "overlap": round(float(overlap_scores[i]), 6),
                "pcd_bonus": round(
                    float(settings.PCD_BONUS_EXCLUSIVE) if vagas[i].is_pcd_exclusive and candidato_pcd else 0.0,
                    6,
                ),
                "score_final": round(float(scores[i]), 6),
                "texto_limpo": vagas[i].texto_limpo[:280],
            }
            for i in top_indices
        ]

    return response


# ── Vaga → Candidatos ────────────────────────────────────────────────────────

async def recomendar_candidatos(
    vaga_id: int,
    top_k: int,
    aplicar_filtro_pcd: bool,
    db: AsyncSession,
    debug: bool = False,
) -> dict:
    """
    Dada uma vaga, retorna os top_k candidatos mais compatíveis.
    filtro_pcd:
      - Se a vaga é PCD exclusiva e filtro=True  → retorna apenas candidatos PCD
      - Se a vaga é PCD exclusiva e filtro=False → retorna base geral (sem restrição)
      - Se vaga ampla e filtro=True              → retorna apenas candidatos PCD
      - Se vaga ampla e filtro=False             → retorna base geral
    """
    vaga_row = await db.get(VagaEmbedding, vaga_id)
    if vaga_row is None:
        return {"error": f"Vaga {vaga_id} não encontrada ou sem embedding gerado."}

    emb_vaga = _deserialize(vaga_row.embedding)
    exige_pcd = vaga_row.is_pcd_exclusive
    vaga_tokens = _tokenize_for_overlap(
        " ".join(
            part for part in (vaga_row.vaga_titulo, vaga_row.vaga_area, vaga_row.texto_limpo) if part
        )
    )

    # Decide o pool de candidatos
    apenas_pcd = aplicar_filtro_pcd
    candidatos = await _load_embeddings_candidatos(db, apenas_pcd=apenas_pcd)

    if not candidatos:
        return {
            "vaga_id": vaga_id,
            "vaga_titulo": vaga_row.vaga_titulo,
            "filtro_pcd_aplicado": aplicar_filtro_pcd,
            "total_retornado": 0,
            "candidatos": [],
        }

    emb_cands = np.stack([_deserialize(c.embedding) for c in candidatos])
    cosine_scores = emb_cands @ emb_vaga

    overlap_scores = np.array(
        [
            _overlap_ratio(vaga_tokens, _tokenize_for_overlap(c.texto_limpo))
            for c in candidatos
        ],
        dtype=np.float32,
    )

    scores = _combined_score(cosine_scores, overlap_scores)

    top_indices = np.argsort(scores)[::-1][:top_k]

    response = {
        "vaga_id":             vaga_id,
        "vaga_titulo":         vaga_row.vaga_titulo,
        "is_pcd_exclusive":    bool(exige_pcd),
        "filtro_pcd_aplicado": aplicar_filtro_pcd,
        "total_retornado":     len(top_indices),
        "candidatos": [
            {
                "cand_id": candidatos[i].cand_id,
                "is_pcd":  bool(candidatos[i].is_pcd),
                "score":   round(float(scores[i]), 4),
            }
            for i in top_indices
        ],
    }

    if debug:
        response["debug"] = [
            {
                "cand_id": candidatos[i].cand_id,
                "is_pcd": bool(candidatos[i].is_pcd),
                "cosine": round(float(cosine_scores[i]), 6),
                "overlap": round(float(overlap_scores[i]), 6),
                "score_final": round(float(scores[i]), 6),
                "texto_limpo": candidatos[i].texto_limpo[:280],
            }
            for i in top_indices
        ]

    return response
