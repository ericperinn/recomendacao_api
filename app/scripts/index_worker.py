import asyncio
import json
from typing import Any

import aio_pika
import numpy as np

from app.core.config import settings
from app.core.database import AsyncSessionLocalEmbeddings, engine_embeddings
from app.models.schema import BaseEmbeddings, CandidatoEmbedding, VagaEmbedding
from app.services.embedder import EmbedderService
from app.services.text_builder import limpar_texto


def _join_parts(parts: list[str]) -> str:
    return limpar_texto(" ".join(part for part in parts if part and str(part).strip()))


def _expand_short_vaga_text(
    titulo: str,
    funcao: str,
    habilidades: list[str],
    conhecimentos: list[str],
    diferenciais: list[str],
    certificacoes: list[str],
    modalidade: str,
    escolaridade_desejada: str,
    area: str,
    area_livre: str,
    descricao: str,
) -> str:
    descricao_base = str(descricao or "").strip()
    if len(descricao_base) >= 180:
        return descricao_base

    termos = [
        *[str(item).strip() for item in habilidades[:4] if str(item).strip()],
        *[str(item).strip() for item in conhecimentos[:3] if str(item).strip()],
        *[str(item).strip() for item in diferenciais[:2] if str(item).strip()],
        *[str(item).strip() for item in certificacoes[:2] if str(item).strip()],
    ]

    contexto = " ".join(
        part
        for part in [
            f"a funcao esta alinhada a {funcao or titulo}",
            f"area {area_livre or area}" if (area_livre or area) else "",
            f"escolaridade desejada {escolaridade_desejada}" if escolaridade_desejada else "",
            f"modalidade {modalidade}" if modalidade else "",
            f"requer {', '.join(termos)}" if termos else "",
        ]
        if part
    )

    bloco_extra = (
        "Atividades e contexto: a pessoa contratada atuara no escopo da vaga, "
        "executando rotinas relacionadas ao cargo, colaborando com o time e mantendo foco em qualidade."
    )

    if contexto:
        bloco_extra = f"{bloco_extra} {contexto}."

    texto_expandido = f"{descricao_base} {bloco_extra}".strip()
    return texto_expandido[:800]


def _build_candidato_text(payload: dict[str, Any]) -> str:
    partes: list[str] = []
    habilidades = payload.get("habilidades") or []
    experiencias = payload.get("experiencias") or []
    formacoes = payload.get("formacoes") or []
    deficiencias = payload.get("deficiencias") or []
    cidade = payload.get("cidade", "")
    estado = payload.get("estado", "")

    if habilidades:
        partes.append("Habilidades: " + "; ".join(habilidades))
    if experiencias:
        partes.append("Experiencia em: " + "; ".join(experiencias))
    if formacoes:
        partes.append("Formacao: " + "; ".join(formacoes))
    if deficiencias:
        partes.append("PCD: " + "; ".join(deficiencias))

    partes.append(f"Local: {cidade} {estado}".strip())
    return _join_parts(partes)


def _build_vaga_text(payload: dict[str, Any]) -> str:
    partes: list[str] = []

    titulo = str(payload.get("titulo", "")).strip()
    area = str(payload.get("area", "")).strip()
    area_livre = str(payload.get("area_livre", "")).strip()
    habilidades = payload.get("habilidades") or []
    conhecimentos = payload.get("conhecimentos") or []
    diferenciais = payload.get("diferenciais") or []
    certificacoes = payload.get("certificacoes") or []
    funcao = str(payload.get("funcao", "")).strip()
    escolaridade_desejada = str(payload.get("escolaridade_desejada", "")).strip()
    modalidade = str(payload.get("modalidade", "")).strip()
    descricao = str(payload.get("descricao", "")).strip()

    if titulo:
        partes.append(f"Vaga: {titulo}")
    if habilidades:
        partes.append("Requisitos: " + "; ".join(habilidades))
    if conhecimentos:
        partes.append("Conhecimentos: " + "; ".join(conhecimentos))
    if diferenciais:
        partes.append("Diferenciais: " + "; ".join(diferenciais))
    if certificacoes:
        partes.append("Certificacoes: " + "; ".join(certificacoes))
    if funcao:
        partes.append(f"Funcao: {funcao}")
    if area_livre:
        partes.append(f"Area: {area_livre}")
    elif area:
        partes.append(f"Area: {area}")
    if escolaridade_desejada:
        partes.append(f"Escolaridade desejada: {escolaridade_desejada}")
    if modalidade:
        partes.append(f"Modalidade: {modalidade}")

    detalhes = _expand_short_vaga_text(
        titulo=titulo,
        funcao=funcao,
        habilidades=[str(item) for item in habilidades],
        conhecimentos=[str(item) for item in conhecimentos],
        diferenciais=[str(item) for item in diferenciais],
        certificacoes=[str(item) for item in certificacoes],
        modalidade=modalidade,
        escolaridade_desejada=escolaridade_desejada,
        area=area,
        area_livre=area_livre,
        descricao=descricao,
    )
    if detalhes:
        partes.append(f"Detalhes: {detalhes}")

    cidade = str(payload.get("cidade", "")).strip()
    estado = str(payload.get("estado", "")).strip()
    partes.append(f"Local: {cidade} {estado}".strip())
    return _join_parts(partes)


async def _process_job(job: dict[str, Any]) -> None:
    entidade = job.get("entidade")
    payload = job.get("payload") or {}
    embedder = EmbedderService.get()

    async with AsyncSessionLocalEmbeddings() as db:
        if entidade == "candidato":
            cand_id = int(payload["cand_id"])
            texto_limpo = _build_candidato_text(payload)
            emb = embedder.encode([texto_limpo])[0]

            existing = await db.get(CandidatoEmbedding, cand_id)
            if existing:
                existing.texto_limpo = texto_limpo
                existing.is_pcd = bool(payload.get("is_pcd", False))
                existing.embedding = emb.astype(np.float32).tobytes()
                existing.modelo_usado = embedder.model_path
            else:
                db.add(
                    CandidatoEmbedding(
                        cand_id=cand_id,
                        texto_limpo=texto_limpo,
                        is_pcd=bool(payload.get("is_pcd", False)),
                        embedding=emb.astype(np.float32).tobytes(),
                        modelo_usado=embedder.model_path,
                    )
                )

        elif entidade == "vaga":
            vaga_id = int(payload["vaga_id"])
            texto_limpo = _build_vaga_text(payload)
            emb = embedder.encode([texto_limpo])[0]

            existing = await db.get(VagaEmbedding, vaga_id)
            if existing:
                existing.vaga_titulo = payload.get("titulo", "")
                existing.vaga_area = payload.get("area", "")
                existing.texto_limpo = texto_limpo
                existing.is_pcd_exclusive = bool(payload.get("is_pcd_exclusive", False))
                existing.embedding = emb.astype(np.float32).tobytes()
                existing.modelo_usado = embedder.model_path
            else:
                db.add(
                    VagaEmbedding(
                        vaga_id=vaga_id,
                        vaga_titulo=payload.get("titulo", ""),
                        vaga_area=payload.get("area", ""),
                        texto_limpo=texto_limpo,
                        is_pcd_exclusive=bool(payload.get("is_pcd_exclusive", False)),
                        embedding=emb.astype(np.float32).tobytes(),
                        modelo_usado=embedder.model_path,
                        vaga_id_empresa=int(payload.get("vaga_id_empresa", 0)),
                    )
                )
        else:
            raise ValueError(f"Entidade de indexacao invalida: {entidade}")

        await db.commit()


async def run_worker() -> None:
    async with engine_embeddings.begin() as conn:
        await conn.run_sync(BaseEmbeddings.metadata.create_all)

    max_retries = 10
    connection = None
    for i in range(max_retries):
        try:
            connection = await aio_pika.connect_robust(settings.BROKER_URL)
            break
        except Exception as e:
            if i == max_retries - 1:
                print("Falha ao conectar no RabbitMQ após várias tentativas.")
                raise
            print(f"Erro ao conectar no RabbitMQ, tentando novamente em 5s... ({e})")
            await asyncio.sleep(5)
            
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=10)
    queue = await channel.declare_queue(settings.INDEX_QUEUE_NAME, durable=True)

    print(f"Worker ouvindo fila '{settings.INDEX_QUEUE_NAME}'")

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process(requeue=True):
                job = json.loads(message.body.decode("utf-8"))
                await _process_job(job)


if __name__ == "__main__":
    asyncio.run(run_worker())
