"""
Endpoints administrativos — protegidos por token simples.
Produção: substitua por OAuth2 / JWT integrado ao sistema principal.
"""
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Literal

from app.core.database import get_db_dados, get_db_embeddings
from app.core.config import settings
from app.services.embedder import EmbedderService
from app.services.infra_status import (
    check_db_dados,
    check_db_embeddings,
    get_broker_queue_status,
)

router = APIRouter(prefix="/admin", tags=["Admin"])


def verificar_token(x_admin_token: str = Header(...)):
    if x_admin_token != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Token inválido.")


class RecalcularRequest(BaseModel):
    entidade: Literal["candidatos", "vagas", "ambos"] = "ambos"


@router.post(
    "/recalcular-embeddings",
    summary="Regenera embeddings após troca de modelo ou novos dados",
    dependencies=[Depends(verificar_token)],
)
async def recalcular_embeddings(
    body: RecalcularRequest,
    db_dados: AsyncSession = Depends(get_db_dados),
    db_embeddings: AsyncSession = Depends(get_db_embeddings),
):
    """
    Regenera todos os embeddings do banco usando o modelo atualmente configurado
    em MODEL_PATH (.env). Use após:
    - Trocar o modelo (base → fine-tuned ou vice-versa)
    - Inserir novos candidatos/vagas em lote
    """
    from app.scripts.seed_and_embed import embed_candidatos, embed_vagas
    from app.scripts.seed_and_embed import _load_csvs

    embedder = EmbedderService.reload()
    resultados = {}
    dfs = _load_csvs()

    if body.entidade in ("candidatos", "ambos"):
        n = await embed_candidatos(db_dados, db_embeddings, embedder, dfs)
        resultados["candidatos_indexados"] = n

    if body.entidade in ("vagas", "ambos"):
        n = await embed_vagas(db_dados, db_embeddings, embedder, dfs)
        resultados["vagas_indexadas"] = n

    return {
        "modelo_usado": embedder.model_path,
        **resultados,
    }


@router.get(
    "/status",
    summary="Status do modelo carregado e contagem de embeddings",
    dependencies=[Depends(verificar_token)],
)
async def status(db: AsyncSession = Depends(get_db_embeddings)):
    from sqlalchemy import func, select
    from app.models.schema import CandidatoEmbedding, VagaEmbedding

    n_cand = (await db.execute(select(func.count()).select_from(CandidatoEmbedding))).scalar()
    n_vaga = (await db.execute(select(func.count()).select_from(VagaEmbedding))).scalar()

    embedder = EmbedderService.get()
    return {
        "modelo_carregado": embedder.model_path,
        "candidatos_indexados": n_cand,
        "vagas_indexadas": n_vaga,
    }


@router.get(
    "/infra-status",
    summary="Saude da infraestrutura (db_dados, db_embeddings, broker e fila)",
    dependencies=[Depends(verificar_token)],
)
async def infra_status():
    db_dados_ok = await check_db_dados()
    db_embeddings_ok = await check_db_embeddings()
    broker = await get_broker_queue_status()

    overall_ok = db_dados_ok and db_embeddings_ok and broker.get("ok", False)

    return {
        "status": "ok" if overall_ok else "degraded",
        "db_dados": "ok" if db_dados_ok else "error",
        "db_embeddings": "ok" if db_embeddings_ok else "error",
        "broker": broker,
    }
