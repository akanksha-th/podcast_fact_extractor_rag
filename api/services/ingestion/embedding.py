from sentence_transformers import SentenceTransformer
import asyncio


class EmbeddingService:
    def __init__(self, model: SentenceTransformer):
        self.embedder = model

    async def generate_embeddings(self, chunks: str):
        embeddings = await asyncio.to_thread(self.embedder.encode, chunks, batch_size=32)
        return embeddings