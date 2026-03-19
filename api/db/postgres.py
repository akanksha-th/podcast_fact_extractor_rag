import asyncpg
from api.core.config import api_settings
from api.utils.security import hash_user_id
import datetime

_pool: asyncpg.Pool | None = None
settings = api_settings()

async def create_pool():
    """Called once at app startup (in lifespan)"""
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=settings.postgres_dsn,
        min_size=settings.postgres.min_pool_size,
        max_size=settings.postgres.max_pool_size
    )
    return _pool
    
async def close_pool():
    """Called once at app shutdown"""
    global _pool
    if _pool:
        await _pool.close()
    
def get_pool() -> asyncpg.Pool:
    """Called by helper functions to access the pool"""
    if _pool is None:
        raise RuntimeError("DB pool is not initialized. call `create_pool()` first.")
    return _pool


# Helper Functions

async def fetchrow(query: str, *args):
    """Fetch a single row. Returns None if not found."""
    async with get_pool().acquire() as conn:
        return await conn.fetchrow(query, *args)
    
async def fetch(query: str, *args):
    """Fetch multiple rows. Returns: list"""
    async with get_pool().acquire() as conn:
        return await conn.fetch(query, *args)
    
async def fetchval(query: str, *args):
    """Fetch a single value."""
    async with get_pool().acquire() as conn:
        return await conn.fetchval(query, *args)
    
async def execute(query: str, user_id: str, *args):
    """Run an INSERT, UPDATE, or DELETE. Returns status string."""
    user_hash = hash_user_id(user_id)
    async with get_pool().acquire() as conn:
        return await conn.execute(query, user_hash, *args)
    
async def execute_raw(query: str, *args):
    """Run an INSERT, UPDATE, or DELETE without user_hash."""
    async with get_pool().acquire() as conn:
        return await conn.execute(query, *args)
    
async def execute_many(query: str, args_list: list):
    """Run bulk insert"""
    async with get_pool().acquire() as conn:
        return await conn.executemany(query, args_list)


# Video Count -> Rate Limiting

async def get_daily_video_count(user_id: str) -> int:
    user_hash = hash_user_id(user_id=user_id)
    today = datetime.datetime.now(datetime.timezone.utc).date()
    query = "SELECT video_count FROM daily_video_limit WHERE user_hash = $1 AND count_date = $2;"
    result = await fetchval(query, user_hash, today)
    return result or 0

async def increment_daily_video_count(user_id: str):
    query = "INSERT INTO daily_video_limit (user_hash, count_date, video_count) \
        VALUES ($1, CURRENT_DATE, 1) \
        ON CONFLICT (user_hash, count_date) \
        DO UPDATE SET video_count = daily_video_limit.video_count + 1;"
    return await execute(query, user_id)
    