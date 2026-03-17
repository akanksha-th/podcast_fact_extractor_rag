from fastapi import APIRouter
from api.db.redis import get_history, delete_session
router = APIRouter()

@router.get("/history")
async def history(user_id: str):
    results = await get_history(user_id=user_id)
    return {"user_id": user_id, "history": results}

@router.delete("/history")
async def delete_session_history(user_id: str):
    await delete_session(user_id=user_id)
    return {"status": "cleared", "message": "Session cleared successfully."}