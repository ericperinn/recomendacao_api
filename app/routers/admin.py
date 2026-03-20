"""
Endpoints administrativos — protegidos por token simples.
Produção: substitua por OAuth2 / JWT integrado ao sistema principal.
"""
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Literal

from app.core.database import get_db
from app.core.config import settings
from app.services.embedder import EmbedderService

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
    db: AsyncSession = Depends(get_db),
):
    """
    Regenera todos os embeddings do banco usando o modelo atualmente configurado
    em MODEL_PATH (.env). Use após:
    - Trocar o modelo (base → fine-tuned ou vice-versa)
    - Inserir novos candidatos/vagas em lote
    """
    from app.scripts.seed_and_embed import embed_candidatos, embed_vagas

    embedder = EmbedderService.reload()
    resultados = {}

    if body.entidade in ("candidatos", "ambos"):
        n = await embed_candidatos(db, embedder)
        resultados["candidatos_indexados"] = n

    if body.entidade in ("vagas", "ambos"):
        n = await embed_vagas(db, embedder)
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
async def status(db: AsyncSession = Depends(get_db)):
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
