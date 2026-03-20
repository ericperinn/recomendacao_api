"""
Fixtures compartilhadas entre todos os testes.
Usa SQLite em memória — sem precisar de Postgres rodando.
O modelo SBERT é mockado para os testes de API serem rápidos.
"""
import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.core.database import Base, get_db
from app.main import app
from app.models.schema import CandidatoEmbedding, VagaEmbedding

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def db_session():
    """Banco SQLite em memória isolado por teste."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


def _make_embedding(seed: int = 0, dim: int = 768) -> bytes:
    """Gera embedding sintético normalizado para testes."""
    rng = np.random.default_rng(seed)
    v   = rng.random(dim).astype(np.float32)
    v  /= np.linalg.norm(v)
    return v.tobytes()


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession):
    """Cliente HTTP com banco e embedder mockados."""

    # Injeta o banco de teste
    app.dependency_overrides[get_db] = lambda: db_session

    # Insere candidatos de teste
    db_session.add(CandidatoEmbedding(
        cand_id=1,
        texto_limpo="desenvolvedor python fastapi",
        is_pcd=False,
        embedding=_make_embedding(1),
        modelo_usado="mock",
    ))
    db_session.add(CandidatoEmbedding(
        cand_id=2,
        texto_limpo="motorista cnh pcd mobilidade reduzida",
        is_pcd=True,
        embedding=_make_embedding(2),
        modelo_usado="mock",
    ))

    # Insere vagas de teste
    db_session.add(VagaEmbedding(
        vaga_id=100,
        vaga_titulo="Dev Full Stack Python",
        vaga_area="Desenvolvimento",
        texto_limpo="dev full stack python fastapi",
        is_pcd_exclusive=False,
        embedding=_make_embedding(1),   # mesmo seed do candidato 1 → score alto
        modelo_usado="mock",
    ))
    db_session.add(VagaEmbedding(
        vaga_id=200,
        vaga_titulo="Vaga PCD - Motorista",
        vaga_area="Logística",
        texto_limpo="motorista pcd adaptacao veicular",
        is_pcd_exclusive=True,
        embedding=_make_embedding(2),   # mesmo seed do candidato 2 → score alto
        modelo_usado="mock",
    ))
    await db_session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    app.dependency_overrides.clear()
