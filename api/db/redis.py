import redis.asyncio as aioredis
from api.core.config import api_settings

_redis: aioredis.Redis | None = None
settings = api_settings()

async def create_redis():
    """Called once at app startup"""
    global _redis
    _redis = await aioredis.from_url(
        settings.redis.cache_url,
        encoding="utf-8",
        decode_responses=True   # always return strings not bytes
    )

async def close_redis():
    """Called once at app shutdown"""
    global _redis
    if _redis:
        await _redis.aclose()

def get_redis():
    if _redis is None:
        raise RuntimeError("Redis is not initialized. Make sure to run `create_redis()` first")
    return _redis


# Helper Functions
## get, set, delete methods and history

async def set_session_field(user_id: str, field: str, value: str) -> None:
    r = get_redis()
    key = f"session:{user_id}:{field}"
    await r.set(key, value, ex=settings.redis.session_ttl)

async def get_session_field(user_id: str, field: str) -> str | None:
    r = get_redis()
    return await r.get(f"session:{user_id}:{field}")

async def delete_session(user_id: str) -> None:
    r = get_redis()
    keys = await r.keys(f"session:{user_id}:*")
    if keys:
        await r.delete(*keys)

async def append_to_history(user_id: str, question: str, answer: str) -> None:
    """Push a QnA pair onto the session history list"""
    r = get_redis()
    key = f"session:{user_id}:history"
    pair= f"{question}|||{answer}"
    await r.rpush(key, pair)
    await r.ltrim(key, -10, -1)     # keep only last 10
    await r.expire(key, settings.redis.session_ttl)     # reset TTL on every new message

async def get_history(user_id: str) -> list[dict]:
    """ Returns last "n" QnA pairs as a list of dicts"""
    r = get_redis()
    raw = await r.lrange(f"session:{user_id}:history", -3, -1)      # last 3 for context window
    result = []
    for pair in raw:
        parts = pair.split("|||", 1)
        if len(parts) == 2:
            result.append({"question": parts[0], "answer": parts[1]})
    return result


## notes caching helpers

async def cache_notes(video_id: str, notes: str) -> None:
    """Cache notes generated and set TTL"""
    r = get_redis()
    await r.set(f"video:{video_id}:notes", notes, ex=settings.redis_session_ttl)

async def get_cached_notes(video_id: str) -> str | None:
    """Returns cached notes or None"""
    r = get_redis()
    return await r.get(f"video:{video_id}:notes")


## rate limiting

async def increment_notes_count(video_id: str, model_version: str, limit: int = 2) -> int:
    r = get_redis()
    key = f"notes_count:{video_id}:{model_version}"
    count = await r.incr(key)

    if count == 1:
        await r.expire(key, 60*60*24*30)    # ttl = 30 days

    if count > limit:
        await r.decr(key)
        raise ValueError(f"Notes rate limit reached for video {video_id}")
    return count