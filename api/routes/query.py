from fastapi import Request, APIRouter, Depends, HTTPException
from api.schema.query_responses import QueryRequest, QueryResponse
from api.services.querying import QueryService
from api.db.redis import get_session_field

router = APIRouter()

def get_query_service(request: Request) -> QueryService:
    return QueryService(request.app.state.embedding_model)

@router.post("/query", response_model=QueryResponse)
async def ask_query(
    body: QueryRequest,
    query_service: QueryService = Depends(get_query_service)
):
    video_id = await get_session_field(body.user_id, field="video_id")
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="Not linked to an active session. Enter URL first."
        )
    
    response = await query_service.get_answer(video_id, body.user_id, body.question)

    return QueryResponse(
        answer=response["result"],
        source_chunks=response["source_chunks"],
        latency_ms=response["latency_ms"]
    )