from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import engine, Base
from app.routers import candidatos, vagas, admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cria tabelas na inicialização (safe — não destrói dados existentes)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
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
