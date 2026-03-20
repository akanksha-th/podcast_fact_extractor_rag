from api.services.ingestion_pipeline import IngestionPipeline
import asyncio
from sentence_transformers import SentenceTransformer
from arq.connections import RedisSettings
from api.core.config import api_settings
from api.db.postgres import create_pool, close_pool
from api.db.redis import create_redis, close_redis, set_session_field
from api.db.qdrant import create_qdrant_client, close_qdrant_client

settings = api_settings()

async def startup(ctx):
    await create_pool()
    await create_redis()
    await create_qdrant_client()
    ctx["model"] = await asyncio.to_thread(
        SentenceTransformer, settings.embedding_model
    )

async def shutdown(ctx):
    await close_qdrant_client()
    await close_redis()
    await close_pool()

async def run_ingestion_job(ctx, video_id: str, video_url: str, user_id: str):
    try:
        model = ctx["model"]
        pipeline = IngestionPipeline(model)
        await pipeline.run_ingestion_pipeline(video_id, video_url, user_id)
    except Exception as e:
        await set_session_field(user_id, "status", "failed")
        raise


class WorkerSettings:
    functions = [run_ingestion_job]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.redis.arq_url)
    max_jobs = 3