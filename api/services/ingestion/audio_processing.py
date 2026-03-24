from groq import AsyncGroq
from pathlib import Path
from api.core.config import api_settings
import os, json
import structlog
import asyncio

settings = api_settings()
logger = structlog.get_logger(__name__)


class AudioTranscriptionService:
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq.api_key)
        self.max_bytes = 24 * 1024 * 1024

    async def get_transcripts(self, audio_path: Path) -> str:
        log = logger.bind(file=audio_path.name)
        if not audio_path.exists():
            log.error("file not found")
            raise FileNotFoundError(f"Audio file {audio_path} does not exist")

        log.info("checking file size...")
        file_size = audio_path.stat().st_size

        if file_size > self.max_bytes:
            log.info("file too large, splitting...", size_mb=file_size / 1e6)
            chunks = await self._split_audio(audio_path)
            
            log.info("transcribing chunks", count=len(chunks))
            transcripts = []
            for i, c in enumerate(chunks):
                if i > 0:
                    await asyncio.sleep(2)
                try:
                    text = await self._transcribe(Path(c))
                    transcripts.append(text)
                except Exception as e:
                    if "429" in str(e):
                        log.error("rate limit hit", detail="ASPH (seconds of audio per hour) limit reached.")
                        raise e
            
            for c in chunks:
                try: Path(c).unlink()
                except Exception: pass
            
            log.info("transcription complete")
            return " ".join(transcripts)
        
        log.info("transcribing_single_file")
        return await self._transcribe(audio_path)

    async def _split_audio(self, audio_path: Path) -> list[Path]:
        probe_cmd = [
            "ffprobe", "-v", "error", 
            "-show_entries", "format=bit_rate:stream=bit_rate",
            "-of", "json", str(audio_path)
        ]
        probe_proc = await asyncio.create_subprocess_exec(
            *probe_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await probe_proc.communicate()
        
        if probe_proc.returncode != 0:
            logger.error("ffprobe_failed", error=stderr.decode())
            raise RuntimeError("Could not determine bitrate.")
        
        bitrate = int(json.loads(stdout)["format"]["bit_rate"])

        segment_seconds = (self.max_bytes*8) / bitrate
        output_pattern = str(audio_path.parent / f"{audio_path.stem}_%03d{audio_path.suffix}")
        ffmpeg_cmd = [
            "ffmpeg", "-i", str(audio_path),
            "-f", "segment",
            "-segment_time", str(segment_seconds),
            "-c", "copy",
            output_pattern
        ]

        logger.info("running ffmpeg split", segment_seconds=round(segment_seconds, 2))
        ffmpeg_proc = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await ffmpeg_proc.wait()

        chunks = sorted(list(audio_path.parent.glob(f"{audio_path.stem}_[0-9][0-9][0-9]{audio_path.suffix}")))
        logger.info("split complete", chunk_count=len(chunks))
        return chunks
    
    async def _transcribe(self, audio_path: Path) -> str:
        logger.info("api request sent", chunk=audio_path.name)
        with open(audio_path, "rb") as f:
            transcription = await self.client.audio.translations.create(
                file=f,
                model="whisper-large-v3",
                response_format="text",
                temperature=0.0
            )
        return transcription.text


async def main():
    audio_path = Path(".audio/8liEuoJA_gc.webm")
    service = AudioTranscriptionService()
    transcripts = await service.get_transcripts(audio_path)
    print(transcripts)

if __name__ == "__main__":
    asyncio.run(main())
