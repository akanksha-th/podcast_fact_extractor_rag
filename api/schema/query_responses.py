from pydantic import BaseModel, field_validator


class QueryRequest(BaseModel):
    user_id: str
    question: str

    @field_validator("question")
    @classmethod
    def question_should_not_be_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user must ask a question")
        return v


class QueryResponse(BaseModel):
    answer: str
    source_chunks: list[dict]
    latency_ms: int | None = None
