from api.db.qdrant import search_chunks
from api.db.redis import append_to_history, get_history
from sentence_transformers import SentenceTransformer
from api.services.ingestion.embedding import EmbeddingService
from api.services.llm_service import QueryLLMService
import structlog
from fastapi import HTTPException

logger = structlog.get_logger(__name__)


class QueryService:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    async def get_answer(self, video_id: str, user_id: str, question: str) -> list[dict]:
        logger.info("starting question embedding...", user_id=user_id, video_id=video_id)
        emb = EmbeddingService(self.model)
        que_embedding = await emb.generate_embeddings([question])
        que_embedding = que_embedding[0].tolist()
        logger.info("emebdding generated successfully", user_id=user_id, video_id=video_id)

        logger.info("fetching chunks from Qdrant...", user_id=user_id, video_id=video_id)
        top_chunks = await search_chunks(video_id=video_id, query_vector=que_embedding)
        if not top_chunks:
            logger.error("no active session for this video", user_id=user_id, video_id=video_id)
            raise HTTPException(
                status_code=404,
                detail="No content found for this video. Please re-ingest with /enter_url"
            )
        
        chunk_texts = [chunk.get("text") for chunk in top_chunks]
        sources = [{"chunk_index": chunk["chunk_index"], "score": chunk["score"]} for chunk in top_chunks]
        history = await get_history(user_id)
        
        logger.info("using LLM now...", user_id=user_id, video_id=video_id, usecase="querying")
        llm = QueryLLMService(chunk_texts, history, question)
        try:
            answer = await llm.get_answer()
            logger.info("answer fetched successfully", user_id=user_id, video_id=video_id)
        except Exception as e:
            logger.error("llm call failed", error=str(e), user_id=user_id, video_id=video_id)
            raise HTTPException(status_code=503, detail="LLM service temporarily unavailable")

        await append_to_history(user_id=user_id, question=question, answer=answer)
        logger.info("Completed!!!\n---", user_id=user_id, video_id=video_id)
        
        return {
            "answer" : answer,
            "source_chunks" : sources,
            "latency_ms" : None
        }
    