from pydantic import BaseModel, field_validator
from typing import Literal

class IngestRequest(BaseModel):
    video_url: str
    user_id: str

    @field_validator("video_url")
    @classmethod
    def must_not_be_empty(cls, v: str)-> str:
        v = v.strip()
        if not v:
            raise ValueError("video_url cannot be empty")
        return v
    
class IngestResponse(BaseModel):
    status: Literal["processing", "ready", "already_exists", "failed"]
    video_id: str
    message: str