from api.db.postgres import fetchrow, execute, execute_raw, increment_daily_video_count
from api.db.qdrant import create_collection, collection_exists, upsert_chunks
from api.db.redis import set_session_field
from api.services.ingestion.transcripts import TranscriptionService
from api.services.ingestion.chunking import ChunkingService
from api.services.ingestion.embedding import EmbeddingService
from sentence_transformers import SentenceTransformer
import structlog

logger = structlog.get_logger(__name__)


class IngestionPipeline:
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.fetch_query = """
        SELECT * FROM ingested_videos WHERE video_id = $1 AND status = 'ready';
        """
        self.exec_query = """
        INSERT INTO request_logs (user_hash, command, video_id, success, latency_ms, error_type) 
        VALUES ($1, '/enter_url', $2, $3, $4, $5);
        """
        self.ingest_query = """
        INSERT INTO ingested_videos (video_id, url, status, chunk_count)
        VALUES ($1, $2, 'ready', $3);
        """

    async def run_ingestion_pipeline(self, video_id: str, video_url: str, user_id: str):
        logger.info("embedding started", user_id=user_id, video_id=video_id)
        is_collection = await collection_exists(video_id)
        row = await fetchrow(self.fetch_query,video_id)
        
        if row and is_collection:
            logger.info("already ingested, setting session", user_id=user_id, video_id=video_id)
            await set_session_field(user_id, "video_id", video_id)
            await set_session_field(user_id, "status", "ready")
        elif is_collection and not row:
            logger.info("collection exists but no Postgres row, setting session", user_id=user_id, video_id=video_id)
            await set_session_field(user_id, "video_id", video_id)
            await set_session_field(user_id, "status", "ready")
        else:
            if await self._get_transcriptions(video_url, video_id, user_id):
                logger.info("fresh ingestion, starting session", user_id=user_id, video=video_id)
                await increment_daily_video_count(user_id)
                await set_session_field(user_id, "video_id", video_id)
                await set_session_field(user_id, "status", "ready")
                logger.info("Completed!!!\n---", user_id=user_id, video=video_id)

    async def _get_transcriptions(self, video_url: str, video_id: str, user_id:str) -> str:
        logger.info("fetching transcript", user_id=user_id, video_id=video_id)
        service = TranscriptionService()
        transcripts = await service.get_transcription(video_id=video_id, url=video_url)
        await set_session_field(user_id=user_id, field="transcript", value=transcripts)
        return await self._create_chunks_and_embeddings(video_id=video_id, transcripts=transcripts, user_id=user_id, video_url=video_url)
    
    async def _create_chunks_and_embeddings(self, video_id: str, transcripts: str, user_id: str, video_url: str) -> bool:
        chunking_service = ChunkingService()
        emb_service = EmbeddingService(self.model)

        try:
            logger.info("creating chunks...", user_id=user_id, video_id=video_id)
            chunks = chunking_service.chunk_transcripts(transcripts)

            logger.info("creating embeddings...", user_id=user_id, video_id=video_id)
            embeddings = await emb_service.generate_embeddings(chunks)

            await create_collection(video_id)
            await upsert_chunks(video_id=video_id, chunks=chunks, embeddings=embeddings)
            await execute_raw(self.ingest_query, video_id, video_url, len(chunks))
            await execute(self.exec_query, user_id, video_id, True, None, None)

            logger.info("successfully created and saved chunks and embeddings", user_id=user_id, video_id=video_id)
            return True
        
        except Exception as e: 
            logger.error("ingestion_failed", error=str(e), user_id=user_id, video_id=video_id)
            await set_session_field(user_id, "status", "failed")
            await execute(self.exec_query, user_id, video_id, False, None, type(e).__name__)
            raise RuntimeError(f"Ingestion couldn't be completed: {e}")