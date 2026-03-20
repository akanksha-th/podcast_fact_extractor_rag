from fastapi import Request, APIRouter, HTTPException, Depends, BackgroundTasks
from api.schema.api_responses import IngestRequest, IngestResponse
from api.utils.url_validator import YouTubeService
from api.services.ingestion_pipeline import IngestionPipeline
from api.db.postgres import get_daily_video_count
from api.db.redis import get_session_field
from api.core.config import api_settings
from arq import create_pool
from arq.connections import RedisSettings

router = APIRouter()
settings = api_settings()

def get_yt_service() -> YouTubeService:
    return YouTubeService()

def get_ingestion_pipeline(request: Request) -> IngestionPipeline:
    return IngestionPipeline(request.app.state.embedding_model)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_url(
    request: Request,
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    yt_service: YouTubeService = Depends(get_yt_service),
    ingestion: IngestionPipeline = Depends(get_ingestion_pipeline)
):
    
    video_id = yt_service.extract_video_id(body.video_url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL"
        )
    
    video_count = await get_daily_video_count(body.user_id)
    if video_count >= settings.max_videos_per_user_per_day:
        raise HTTPException(
            status_code=429,
            detail="Rate limit for today has been exhausted"
        )
    
    # background_tasks.add_task(
    #     ingestion.run_ingestion_pipeline, video_id, body.video_url, body.user_id)
    # status = await get_session_field(body.user_id, "status")


    arq_pool = request.app.state.arq_pool
    await arq_pool.enqueue_job(
        "run_ingestion_job", video_id, body.video_url, body.user_id
    )

    return IngestResponse(
        status="processing",
        video_id=video_id,
        message="⏳ Processing your podcast..."
    )