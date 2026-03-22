from api.db.redis import cache_notes
from api.db.postgres import execute
from api.db.qdrant import get_all_chunks
from api.core.config import api_settings
from api.services.llm_service import NotesLLMService
import structlog

logger = structlog.get_logger(__name__)
settings = api_settings()


class NotesGenService:
    def __init__(self):
        self.llm = NotesLLMService()
        self.model_version = settings.groq.notes_model
        self.exec_query = """
        INSERT INTO notes_generation_log (user_hash, video_id, notes_content, model_version)
        VALUES ($1, $2, $3, $4);"""

    async def get_notes(self, user_id: str, video_id: str) -> str:
        logger.info("getting chunk batches...", user_id=user_id, video_id=video_id)
        chunks = await get_all_chunks(video_id=video_id)      # list[str]
        batches = [chunks[i:i+10] for i in range(0, len(chunks), 10)]
        
        summaries = []
        for batch in batches:
            summary = await self._map("\n".join(batch), user_id=user_id, video_id=video_id)
            summaries.append(summary)
        
        final_summary = await self._reduce(user_id, video_id, "\n".join(summaries))

        logger.info("caching notes...", user_id=user_id, video_id=video_id)
        await cache_notes(video_id=video_id, notes=final_summary)
        logger.info("saving to db...", user_id=user_id, video_id=video_id)
        await execute(self.exec_query, user_id, video_id, final_summary, self.model_version)

        logger.info("Completed!!!\n---", user_id=user_id, video_id=video_id)
        return final_summary

    async def _map(self, chunks: str, user_id: str, video_id: str):
        logger.info("chunk summarization...", user_id=user_id, video_id=video_id)
        return await self.llm.map_summary(chunks)

    async def _reduce(self, user_id: str, video_id: str, summaries: str):
        logger.info("final summarization...", user_id=user_id, video_id=video_id)
        return await self.llm.reduce_summary(summaries)