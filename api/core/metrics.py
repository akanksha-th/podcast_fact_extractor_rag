from prometheus_client import Histogram, Counter

embedding_duration = Histogram(
    "embedding_duration_seconds",
    "Time spent generating embeddings",
    buckets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

groq_latency = Histogram(
    "groq_api_duration_seconds",
    "Time spent on Groq API calls",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

notes_cache_hits = Counter(
    "notes_cache_hits_total",
    "Number of notes cache hits"
)

notes_cache_misses = Counter(
    "notes_cache_misses_total",
    "Number of notes cache misses"
)