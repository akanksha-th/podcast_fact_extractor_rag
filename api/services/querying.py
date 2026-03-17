from api.db.qdrant import search_chunks
from api.db.redis import append_to_history, get_history
from sentence_transformers import SentenceTransformer
from api.services.ingestion.embedding import EmbeddingService
from api.services.llm_service import QueryLLMService


class QueryService:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    async def get_answer(self, video_id: str, user_id: str, question: str) -> list[dict]:
        emb = EmbeddingService(self.model)
        que_embedding = await emb.generate_embeddings([question])
        que_embedding = que_embedding[0].tolist()

        top_chunks = await search_chunks(video_id=video_id, query_vector=que_embedding)
        chunk_texts = [chunk.get("text") for chunk in top_chunks]
        sources = [{"chunk_index": chunk["chunk_index"], "score": chunk["score"]} for chunk in top_chunks]
        history = await get_history(user_id)
        
        llm = QueryLLMService(chunk_texts, history, question)
        answer = await llm.get_answer()

        await append_to_history(user_id=user_id, question=question, answer=answer)
        
        return {
            "result" : answer,
            "source_chunks" : sources,
            "latency_ms" : None
        }
    