from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.config import settings
from app.services.recommender import recomendar_vagas

router = APIRouter(prefix="/candidatos", tags=["Candidatos"])


# ── Recomendação ─────────────────────────────────────────────────────────────

@router.get(
    "/{cand_id}/vagas",
    summary="Vagas recomendadas para um candidato",
)
async def get_vagas_recomendadas(
    cand_id: int,
    top_k: int = Query(default=None, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """
    Retorna as vagas mais compatíveis com o perfil do candidato.

    - Candidatos **PCD** recebem bônus em vagas exclusivas PCD.
    - Candidatos **não-PCD** não veem vagas exclusivas PCD.
    - `score` entre 0 e 1 — quanto maior, melhor o match.
    """
    result = await recomendar_vagas(cand_id, top_k or settings.TOP_K_DEFAULT, db)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Upsert individual ─────────────────────────────────────────────────────────

class CandidatoUpsertRequest(BaseModel):
    """
    Payload para indexar (ou reindexar) um único candidato sem reprocessar toda a base.
    Útil quando o sistema principal cria ou atualiza um candidato.
    """
    cand_id: int
    nome: str
    cidade: str
    estado: str
    is_pcd: bool = False
    habilidades: list[str] = []
    experiencias: list[str] = []    # lista de cargos
    formacoes: list[str] = []       # lista de cursos
    deficiencias: list[str] = []    # lista de descrições de deficiência


@router.post(
    "/indexar",
    summary="Indexa ou reindexia um único candidato",
    status_code=201,
)
async def indexar_candidato(
    body: CandidatoUpsertRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Monta o texto semântico, gera o embedding e salva/atualiza em
    `candidato_embeddings`. Use quando o sistema principal criar ou editar um candidato.
    """
    from app.services.text_builder import limpar_texto
    from app.services.embedder import EmbedderService
    from app.models.schema import CandidatoEmbedding
    import numpy as np

    partes = []
    if body.habilidades:
        partes.append("Habilidades: " + "; ".join(body.habilidades))
    if body.experiencias:
        partes.append("Experiência em: " + "; ".join(body.experiencias))
    if body.formacoes:
        partes.append("Formação: " + "; ".join(body.formacoes))
    if body.deficiencias:
        partes.append("PCD: " + "; ".join(body.deficiencias))
    partes.append(f"Local: {body.cidade} {body.estado}")

    texto_limpo = limpar_texto(" ".join(partes))
    embedder    = EmbedderService.get()
    emb         = embedder.encode([texto_limpo])[0]

    existing = await db.get(CandidatoEmbedding, body.cand_id)
    if existing:
        existing.texto_limpo  = texto_limpo
        existing.is_pcd       = body.is_pcd
        existing.embedding    = emb.astype(np.float32).tobytes()
        existing.modelo_usado = embedder.model_path
        acao = "atualizado"
    else:
        db.add(CandidatoEmbedding(
            cand_id=body.cand_id,
            texto_limpo=texto_limpo,
            is_pcd=body.is_pcd,
            embedding=emb.astype(np.float32).tobytes(),
            modelo_usado=embedder.model_path,
        ))
        acao = "criado"

    await db.commit()
    return {
        "cand_id":      body.cand_id,
        "texto_gerado": texto_limpo,
        "modelo_usado": embedder.model_path,
        "acao":         acao,
    }
