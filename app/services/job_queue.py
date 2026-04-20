import json
from typing import Any

import aio_pika

from app.core.config import settings


async def enqueue_index_job(job: dict[str, Any]) -> None:
    """Publica um job de indexacao para processamento assíncrono no worker."""
    connection = await aio_pika.connect_robust(settings.BROKER_URL)
    try:
        channel = await connection.channel()
        queue = await channel.declare_queue(settings.INDEX_QUEUE_NAME, durable=True)
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(job).encode("utf-8"),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                content_type="application/json",
            ),
            routing_key=queue.name,
        )
    finally:
        await connection.close()
