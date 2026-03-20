from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.config import settings
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
    db: AsyncSession = Depends(get_db),
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
    result = await recomendar_candidatos(vaga_id, top_k or settings.TOP_K_DEFAULT, filtrar_pcd, db)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Upsert individual ─────────────────────────────────────────────────────────

class VagaUpsertRequest(BaseModel):
    """
    Payload para indexar (ou reindexar) uma única vaga sem reprocessar toda a base.
    """
    vaga_id: int
    titulo: str
    cidade: str
    estado: str
    area: str = ""
    is_pcd_exclusive: bool = False
    habilidades: list[str] = []
    conhecimentos: list[str] = []
    funcao: str = ""
    descricao: str = ""


@router.post(
    "/indexar",
    summary="Indexa ou reindexia uma única vaga",
    status_code=201,
)
async def indexar_vaga(
    body: VagaUpsertRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Monta o texto semântico, gera o embedding e salva/atualiza em
    `vaga_embeddings`. Use quando o sistema principal criar ou editar uma vaga.
    """
    from app.services.text_builder import limpar_texto
    from app.services.embedder import EmbedderService
    from app.models.schema import VagaEmbedding
    import numpy as np

    partes = [f"Vaga: {body.titulo}"]
    if body.habilidades:
        partes.append("Requisitos: " + "; ".join(body.habilidades))
    if body.conhecimentos:
        partes.append("Conhecimentos: " + "; ".join(body.conhecimentos))
    if body.funcao:
        partes.append(f"Função: {body.funcao}")
    if body.descricao:
        partes.append(f"Detalhes: {body.descricao[:400]}")
    partes.append(f"Local: {body.cidade} {body.estado}")

    texto_limpo = limpar_texto(" ".join(partes))
    embedder    = EmbedderService.get()
    emb         = embedder.encode([texto_limpo])[0]

    existing = await db.get(VagaEmbedding, body.vaga_id)
    if existing:
        existing.vaga_titulo      = body.titulo
        existing.vaga_area        = body.area
        existing.texto_limpo      = texto_limpo
        existing.is_pcd_exclusive = body.is_pcd_exclusive
        existing.embedding        = emb.astype(np.float32).tobytes()
        existing.modelo_usado     = embedder.model_path
        acao = "atualizado"
    else:
        db.add(VagaEmbedding(
            vaga_id=body.vaga_id,
            vaga_titulo=body.titulo,
            vaga_area=body.area,
            texto_limpo=texto_limpo,
            is_pcd_exclusive=body.is_pcd_exclusive,
            embedding=emb.astype(np.float32).tobytes(),
            modelo_usado=embedder.model_path,
        ))
        acao = "criado"

    await db.commit()
    return {
        "vaga_id":      body.vaga_id,
        "texto_gerado": texto_limpo,
        "modelo_usado": embedder.model_path,
        "acao":         acao,
    }
