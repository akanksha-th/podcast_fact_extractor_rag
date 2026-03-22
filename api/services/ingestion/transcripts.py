import yt_dlp
from faster_whisper import WhisperModel
import asyncio
import httpx
import os
import torch

class TranscriptionService:
    async def _parse_yt(self, data: dict) -> str:
        """
        Converts Youtube json subtitle format into plain text.
        """
        events = data.get("events") or []
        lines = [
            s.get("utf8", "").strip()
            for ev in events 
            for s in ev.get("segs", [])
        ]

        cleaned_text = " ".join(t for t in lines if t)
        return cleaned_text

    async def get_transcription(self, video_id: str, url: str):
        """
        1. Tries fetching YouTube subtitles for transcription.
        2. If unavailable, downloads audio and transcribes.
        """
        try:
            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitlesformat": "json3",
                "skip_download": True,
                "subtitleslangs": ["en"],
                "quiet": True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await asyncio.to_thread(ydl.extract_info, url, False)
                subs = info.get("subtitles") or info.get("automatic_captions")

                if subs and "en" in subs:
                    transcript_url = subs["en"][0]["url"]

                    async with httpx.AsyncClient() as client:
                        json_data = await client.get(transcript_url)
                    new_data = json_data.json()
                    return await self._parse_yt(new_data)
            
        except Exception as e:
            pass

        try:
            os.makedirs(".audio", exist_ok=True)
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": f".audio/{video_id}.%(ext)s",
                "quiet": True,
                'no_warnings': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await asyncio.to_thread(ydl.extract_info, url, True)
                audio_path = ydl.prepare_filename(info)

        except Exception as e:
            raise RuntimeError(f"Audio download failed: {e}")

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = await asyncio.to_thread(WhisperModel, "small", device=device, compute_type=compute_type)
        try:
            segments, _ = await asyncio.to_thread(model.transcribe, audio_path, task="translate", language="en", log_progress=True)
            text = "\n".join([s.text for s in segments])
            return text
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

