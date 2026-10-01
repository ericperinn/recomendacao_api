import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import engine_dados, engine_embeddings
from app.models.schema import BaseDados, BaseEmbeddings
from app.routers import candidatos, vagas, admin

logger = logging.getLogger("uvicorn.error")


async def _initialize_databases(max_attempts: int = 10, delay_seconds: float = 2.0) -> None:
    """Cria schema com retry para tolerar atrasos na subida dos bancos."""
    for attempt in range(1, max_attempts + 1):
        try:
            async with engine_dados.begin() as conn:
                await conn.run_sync(BaseDados.metadata.create_all)

            async with engine_embeddings.begin() as conn:
                await conn.run_sync(BaseEmbeddings.metadata.create_all)

            if attempt > 1:
                logger.info("Conexao com bancos restabelecida na tentativa %s.", attempt)
            return
        except OSError as exc:
            if attempt == max_attempts:
                raise

            logger.warning(
                "Falha ao conectar nos bancos (tentativa %s/%s): %s. Nova tentativa em %.1fs.",
                attempt,
                max_attempts,
                exc,
                delay_seconds,
            )
            await asyncio.sleep(delay_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cria tabelas na inicializacao (safe — nao destroi dados existentes).
    await _initialize_databases()
    yield


app = FastAPI(
    title="API de Recomendação — TCC",
    description=(
        "Recomendação semântica de vagas e candidatos usando SBERT + similaridade do cosseno. "
        "Suporta filtro PCD para vagas exclusivas."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrinja para o domínio do sistema principal em produção
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(candidatos.router)
app.include_router(vagas.router)
app.include_router(admin.router)


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
