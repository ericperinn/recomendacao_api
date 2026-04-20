from sqlalchemy import text
import aio_pika

from app.core.config import settings
from app.core.database import engine_dados, engine_embeddings


async def check_db_dados() -> bool:
    try:
        async with engine_dados.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def check_db_embeddings() -> bool:
    try:
        async with engine_embeddings.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def get_broker_queue_status() -> dict:
    try:
        connection = await aio_pika.connect_robust(settings.BROKER_URL)
        try:
            channel = await connection.channel()
            queue = await channel.declare_queue(
                settings.INDEX_QUEUE_NAME,
                durable=True,
            )
            return {
                "ok": True,
                "queue": settings.INDEX_QUEUE_NAME,
                "messages": queue.declaration_result.message_count,
                "consumers": queue.declaration_result.consumer_count,
            }
        finally:
            await connection.close()
    except Exception as exc:
        return {
            "ok": False,
            "queue": settings.INDEX_QUEUE_NAME,
            "error": str(exc),
        }
