from sentence_transformers import SentenceTransformer
import asyncio
from api.core.metrics import embedding_duration
import time


class EmbeddingService:
    def __init__(self, model: SentenceTransformer):
        self.embedder = model

    async def generate_embeddings(self, chunks: str):
        start = time.time()
        embeddings = await asyncio.to_thread(self.embedder.encode, chunks, batch_size=32)
        embedding_duration.observe(time.time() - start)
        return embeddings