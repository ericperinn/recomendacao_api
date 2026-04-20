from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

engine_dados = create_async_engine(settings.DATABASE_URL_DADOS, echo=False)
engine_embeddings = create_async_engine(settings.DATABASE_URL_EMBEDDINGS, echo=False)

AsyncSessionLocalDados = async_sessionmaker(engine_dados, expire_on_commit=False)
AsyncSessionLocalEmbeddings = async_sessionmaker(engine_embeddings, expire_on_commit=False)


class BaseDados(DeclarativeBase):
    pass


class BaseEmbeddings(DeclarativeBase):
    pass


async def get_db_dados() -> AsyncSession:
    async with AsyncSessionLocalDados() as session:
        yield session


async def get_db_embeddings() -> AsyncSession:
    async with AsyncSessionLocalEmbeddings() as session:
        yield session


# Compatibilidade temporária para módulos legados.
engine = engine_dados
AsyncSessionLocal = AsyncSessionLocalDados
Base = BaseDados
get_db = get_db_embeddings
