"""
Lógica central de recomendação.
Busca embeddings pré-calculados do banco e computa similaridade do cosseno.
Como os embeddings já estão normalizados (L2), produto escalar == cosine similarity.
"""
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema import CandidatoEmbedding, VagaEmbedding
from app.core.config import settings

EMBEDDING_DIM = 768  # paraphrase-multilingual-mpnet-base-v2


def _deserialize(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _serialize(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


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

    emb_cand     = _deserialize(cand_row.embedding)
    candidato_pcd = cand_row.is_pcd

    # Busca todos embeddings de vagas
    vagas = await _load_embeddings_vagas(db)
    if not vagas:
        return {"error": "Nenhuma vaga indexada. Execute o seed primeiro."}

    emb_vagas = np.stack([_deserialize(v.embedding) for v in vagas])

    # Produto escalar (== cosine sim porque os vetores estão L2-normalizados)
    scores = emb_vagas @ emb_cand

    # Regras PCD
    for i, vaga in enumerate(vagas):
        if vaga.is_pcd_exclusive:
            if candidato_pcd:
                scores[i] += 0.25   # bônus: sobe vagas PCD no ranking
            else:
                scores[i] = 0.0     # zera: candidato não-PCD não vê vaga exclusiva PCD

    # Ranqueia
    top_indices = np.argsort(scores)[::-1]
    top_indices = [i for i in top_indices if scores[i] > 0][:top_k]

    return {
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


# ── Vaga → Candidatos ────────────────────────────────────────────────────────

async def recomendar_candidatos(
    vaga_id: int,
    top_k: int,
    aplicar_filtro_pcd: bool,
    db: AsyncSession,
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

    emb_vaga   = _deserialize(vaga_row.embedding)
    exige_pcd  = vaga_row.is_pcd_exclusive

    # Decide o pool de candidatos
    apenas_pcd = exige_pcd and aplicar_filtro_pcd or (not exige_pcd and aplicar_filtro_pcd)
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
    scores    = emb_cands @ emb_vaga

    top_indices = np.argsort(scores)[::-1][:top_k]

    return {
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
