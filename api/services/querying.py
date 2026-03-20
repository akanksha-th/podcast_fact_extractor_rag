from api.db.qdrant import search_chunks
from api.db.redis import append_to_history, get_history
from sentence_transformers import SentenceTransformer
from api.services.ingestion.embedding import EmbeddingService
from api.services.llm_service import QueryLLMService
import logging
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryService:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    async def get_answer(self, video_id: str, user_id: str, question: str) -> list[dict]:
        logger.info("Starting question embedding")
        emb = EmbeddingService(self.model)
        que_embedding = await emb.generate_embeddings([question])
        que_embedding = que_embedding[0].tolist()
        logger.info("Emebdding generated successfully")

        logger.info("Fetching top-k chunks from Qdrant")
        top_chunks = await search_chunks(video_id=video_id, query_vector=que_embedding)
        if not top_chunks:
            raise HTTPException(
                status_code=404,
                detail="No content found for this video. Please re-ingest with /enter_url"
            )
        
        chunk_texts = [chunk.get("text") for chunk in top_chunks]
        # logger.info(f"Chunk texts: {chunk_texts}")
        sources = [{"chunk_index": chunk["chunk_index"], "score": chunk["score"]} for chunk in top_chunks]
        history = await get_history(user_id)
        
        logger.info("Using LLM now...")
        llm = QueryLLMService(chunk_texts, history, question)
        answer = await llm.get_answer()
        logger.info("Completed!!!")

        await append_to_history(user_id=user_id, question=question, answer=answer)
        
        return {
            "answer" : answer,
            "source_chunks" : sources,
            "latency_ms" : None
        }
    