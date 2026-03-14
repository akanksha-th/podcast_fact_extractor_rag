from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from api.core.config import api_settings

_client: AsyncQdrantClient | None = None

settings = api_settings()
DISTANCE = Distance.COSINE


async def create_qdrant_client() -> None:
    """Called once at app startup"""
    global _client
    _client = AsyncQdrantClient(url=settings.qdrant_url)

async def close_qdrant_client() -> None:
    global _client
    if _client:
        await _client.close()

def get_client() -> AsyncQdrantClient:
    if _client is None:
        raise RuntimeError("Qdrant client not initialized.")
    return _client


# Helper Functions

async def collection_exists(video_id: str) -> bool:
    client = get_client()
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]
    return video_id in names

async def create_collection(video_id: str) -> None:
    """Create a new collection for a video. One collection = one podcast."""
    client = get_client()
    await client.create_collection(
        collection_name=video_id,
        vectors_config=VectorParams(size=settings.qdrant.vector_size, distance=DISTANCE)
    )

async def upsert_chunks(video_id: str, chunks: list[dict]) -> None:
    """Store embedded chunks into Qdrant."""
    client = get_client()
    points = [
        PointStruct(
            id=chunk["id"],
            vector=chunk["vector"],
            payload={
                "text": chunk["text"],
                "chunk_index": chunk["chunk_index"],
                "video_id": video_id
            }
        )
        for chunk in chunks
    ]
    await client.upsert(collection_name=video_id, points=points)


## search and get chunks

async def search_chunks(video_id: str, query_vector: list[float], top_k: int = 5) -> list[dict]:
    """Find top_k most relevant chunks for a query vector and returns list of dicts"""
    client = get_client()
    results = await client.query_points(
        collection_name=video_id,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )
    return [{
        "text": r.payload["text"],
        "chunk_index": r.payload["chunk_index"],
        "score": r.score
    } for r in results.points ]

async def get_all_chunks(video_id: str) -> list[str]:
    """Retrieve ALL chunks from a collection"""
    client = get_client()
    all_texts = []
    offset = None

    while True:
        results, offset = await client.scroll(
            collection_name=video_id,
            limit=100,  # fetch 100 at a time
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        results.sort(key=lambda r: r.payload.get("chunk_size", 0))
        all_texts.extend(r.payload["text"] for r in results)
        if offset is None:
            break

    return all_texts