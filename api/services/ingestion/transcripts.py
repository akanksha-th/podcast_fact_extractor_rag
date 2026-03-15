import yt_dlp
from faster_whisper import WhisperModel
import asyncio
import httpx
import os

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
            output = f"scripts/{video_id}.mp3"
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": output,
                "quiet": True,
                'no_warnings': True,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192"
                }],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                await asyncio.to_thread(ydl.download, [url])

        except Exception as e:
            raise RuntimeError(f"Audio download failed: {e}")

        model = await asyncio.to_thread(WhisperModel, "small", device="cpu", compute_type="int8")
        try:
            segments, _ = await asyncio.to_thread(model.transcribe, output, task="translate", language="en")
            text = "\n".join([s.text for s in segments])
            return text
        finally:
            if os.path.exists(output):
                os.remove(output)