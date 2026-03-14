from fastapi import FastAPI
from api.routes import ingest

from contextlib import asynccontextmanager
from api.db.postgres import create_pool, close_pool
from api.db.redis import create_redis, close_redis
from api.db.qdrant import create_qdrant_client, close_qdrant_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.pg_pool = await create_pool()
    app.state.redis = await create_redis()
    app.state.qdrant = await create_qdrant_client()
    yield
    # Shutdown
    await close_qdrant_client()
    await close_redis()
    await close_pool()

app = FastAPI(
    title="Podcast-RAG",
    description="Generate notes and ask questions based on a YT video",
    version="1.0.0",
    lifespan=lifespan
    )

@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(ingest.router)


if __name__ == "__main__":
    print(app.state.pg_pool)