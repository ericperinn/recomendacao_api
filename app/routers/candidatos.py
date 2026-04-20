from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete as sql_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_embeddings
from app.core.config import settings
from app.models.schema import CandidatoEmbedding
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
    debug: bool = Query(default=False, description="Inclui scores detalhados por vaga."),
    db: AsyncSession = Depends(get_db_embeddings),
):
    """
    Retorna as vagas mais compatíveis com o perfil do candidato.

    - Candidatos **PCD** recebem bônus em vagas exclusivas PCD.
    - Candidatos **não-PCD** não veem vagas exclusivas PCD.
    - `score` entre 0 e 1 — quanto maior, melhor o match.
    """
    result = await recomendar_vagas(cand_id, top_k or settings.TOP_K_DEFAULT, db, debug=debug)
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
    summary="Enfileira indexação de um único candidato",
    status_code=202,
)
async def indexar_candidato(
    body: CandidatoUpsertRequest,
):
    """
    Enfileira um job de indexação para processamento assíncrono no worker.
    Use quando o sistema principal criar ou editar um candidato.
    """
    from app.services.job_queue import enqueue_index_job

    await enqueue_index_job({
        "entidade": "candidato",
        "payload": body.model_dump(),
    })

    return {
        "cand_id": body.cand_id,
        "status": "enfileirado",
    }


# ── Deleção de embedding ──────────────────────────────────────────────────────

@router.delete(
    "/{cand_id}",
    summary="Remove o embedding de um candidato",
    status_code=200,
)
async def deletar_embedding_candidato(
    cand_id: int,
    db: AsyncSession = Depends(get_db_embeddings),
):
    """
    Remove o embedding pré-calculado do candidato do banco de embeddings.
    Deve ser chamado pelo sistema principal quando um candidato é deletado.
    """
    result = await db.execute(
        sql_delete(CandidatoEmbedding).where(CandidatoEmbedding.cand_id == cand_id)
    )
    await db.commit()

    if result.rowcount == 0:
        # Não é erro crítico — embedding pode nunca ter sido gerado
        return {"cand_id": cand_id, "status": "nao_encontrado"}

    return {"cand_id": cand_id, "status": "removido"}
