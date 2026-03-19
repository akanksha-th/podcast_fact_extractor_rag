from fastapi import APIRouter, Depends, HTTPException
from api.db.postgres import fetchval
from api.db.redis import get_cached_notes, get_session_field
from api.core.config import api_settings
from api.services.notes_service import NotesGenService

router = APIRouter()
settings = api_settings()

def get_notes_service() -> NotesGenService:
    return NotesGenService()

@router.post("/notes")
async def fetch_notes(
    user_id: str,
    notes_service: NotesGenService = Depends(get_notes_service)
    ):
    video_id = await get_session_field(user_id, "video_id")
    if not video_id:
        raise HTTPException(status_code=400, detail="No active session.")
    
    query = """SELECT COUNT(*) FROM notes_generation_log 
        WHERE video_id = $1 AND model_version = $2;"""
    count = await fetchval(query, video_id, settings.groq.model_version)
    if count >= 2:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded."
    )
    
    cached_notes = await get_cached_notes(video_id=video_id)
    if cached_notes:
        return {"notes": cached_notes, "video_id": video_id}

    notes = await notes_service.get_notes(user_id, video_id)
    return {"notes": notes, "video_id": video_id}