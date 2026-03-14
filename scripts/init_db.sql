CREATE TABLE ingested_videos (
    video_id        VARCHAR(20) PRIMARY KEY,
    url             TEXT NOT NULL,
    title           TEXT,
    duration_secs   INTEGER,
    language        VARCHAR(10),
    transcript_hash VARCHAR(64),
    chunk_count     INTEGER,
    status          VARCHAR(20),
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    model_used      VARCHAR(50)
);

CREATE TABLE notes_generation_log (
    id              SERIAL PRIMARY KEY,
    video_id        VARCHAR(20) REFERENCES ingested_videos(video_id),
    user_hash       VARCHAR(64),     --hashed telegram user_id
    notes_content   TEXT,
    model_version   VARCHAR(50) NOT NULL,
    generated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE request_logs (
    id              BIGSERIAL PRIMARY KEY,
    request_id      UUID DEFAULT gen_random_uuid(),
    user_hash       VARCHAR(64),
    command         VARCHAR(30),      --/ask, /get_notes, /enter_url
    video_id        VARCHAR(20),
    success         BOOLEAN,
    latency_ms      INTEGER,
    error_type      VARCHAR(50),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE daily_video_limit (
    user_hash VARCHAR(64),
    count_date DATE NOT NULL,
    video_count INT NOT NULL DEFAULT 0,
    PRIMARY KEY (user_hash, count_date) 
);