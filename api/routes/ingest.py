from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from api.schema.api_responses import IngestRequest, IngestResponse
from api.utils.url_validator import YouTubeService
from api.utils.get_transcripts import TranscriptionService
from api.db.postgres import get_daily_video_count
from api.core.config import api_settings

router = APIRouter()
settings = api_settings()

def get_yt_service() -> YouTubeService:
    return YouTubeService()

def get_transcript_service() -> TranscriptionService:
    return TranscriptionService()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_url(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    yt_service: YouTubeService = Depends(get_yt_service),
    ts_service: TranscriptionService = Depends(get_transcript_service)
):
    # Step 1: check if valid YT URL
    video_id = yt_service.extract_video_id(request.video_url)
    video_count = await get_daily_video_count(request.user_id)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL"
        )
    elif video_count >= settings.max_videos_per_user_per_day:
        raise HTTPException(
            status_code=429,
            detail="Rate limit for today has been exhausted"
        )
    
    # try:...
    
    background_tasks.add_task(
        ts_service.get_transcription, 
        video_id,
        request.video_url,
        request.user_id)

    return IngestResponse(
        status="processing",
        video_id=video_id,
        message="⏳ Processing your podcast..."
    )