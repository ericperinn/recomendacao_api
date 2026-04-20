from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete as sql_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_embeddings
from app.core.config import settings
from app.models.schema import VagaEmbedding
from app.services.recommender import recomendar_candidatos

router = APIRouter(prefix="/vagas", tags=["Vagas"])


# ── Recomendação ─────────────────────────────────────────────────────────────

@router.get(
    "/{vaga_id}/candidatos",
    summary="Candidatos ranqueados para uma vaga",
)
async def get_candidatos_ranqueados(
    vaga_id: int,
    top_k: int = Query(default=None, ge=1, le=100),
    filtrar_pcd: bool = Query(
        default=True,
        description=(
            "True → vagas PCD retornam apenas candidatos PCD. "
            "False → busca na base geral independente da flag PCD."
        ),
    ),
    debug: bool = Query(default=False, description="Inclui scores detalhados por candidato."),
    db: AsyncSession = Depends(get_db_embeddings),
):
    """
    Retorna os candidatos mais compatíveis com a vaga — visão do recrutador.

    | Vaga          | filtrar_pcd | Pool de candidatos         |
    |---------------|-------------|----------------------------|
    | PCD exclusiva | True        | Apenas candidatos PCD      |
    | PCD exclusiva | False       | Base geral                 |
    | Ampla         | True        | Apenas candidatos PCD      |
    | Ampla         | False       | Base geral                 |
    """
    result = await recomendar_candidatos(vaga_id, top_k or settings.TOP_K_DEFAULT, filtrar_pcd, db, debug=debug)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Upsert individual ─────────────────────────────────────────────────────────

class VagaUpsertRequest(BaseModel):
    """
    Payload para indexar (ou reindexar) uma única vaga sem reprocessar toda a base.
    """
    vaga_id: int
    vaga_id_empresa: int = 0
    titulo: str
    cidade: str
    estado: str
    area: str = ""
    area_livre: str = ""
    is_pcd_exclusive: bool = False
    habilidades: list[str] = []
    conhecimentos: list[str] = []
    diferenciais: list[str] = []
    certificacoes: list[str] = []
    funcao: str = ""
    escolaridade_desejada: str = ""
    modalidade: str = ""
    descricao: str = ""


@router.post(
    "/indexar",
    summary="Enfileira indexação de uma única vaga",
    status_code=202,
)
async def indexar_vaga(
    body: VagaUpsertRequest,
):
    """
    Enfileira um job de indexação para processamento assíncrono no worker.
    Use quando o sistema principal criar ou editar uma vaga.
    """
    from app.services.job_queue import enqueue_index_job

    await enqueue_index_job({
        "entidade": "vaga",
        "payload": body.model_dump(),
    })

    return {
        "vaga_id": body.vaga_id,
        "status": "enfileirado",
    }


# ── Deleção de embedding ──────────────────────────────────────────────────────

@router.delete(
    "/{vaga_id}",
    summary="Remove o embedding de uma vaga",
    status_code=200,
)
async def deletar_embedding_vaga(
    vaga_id: int,
    db: AsyncSession = Depends(get_db_embeddings),
):
    """
    Remove o embedding pré-calculado da vaga do banco de embeddings.
    Deve ser chamado pelo sistema principal quando uma vaga é deletada.
    """
    result = await db.execute(
        sql_delete(VagaEmbedding).where(VagaEmbedding.vaga_id == vaga_id)
    )
    await db.commit()

    if result.rowcount == 0:
        return {"vaga_id": vaga_id, "status": "nao_encontrado"}

    return {"vaga_id": vaga_id, "status": "removido"}
