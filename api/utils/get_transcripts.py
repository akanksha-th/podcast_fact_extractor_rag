import yt_dlp
from faster_whisper import WhisperModel


class TranscriptionService:
    def _parse_yt(self, data: dict) -> str:
        """
        Converts Youtube json subtitle format into plain text.
        """
        events = data.get("events")
        # print(len(events))
        lines = []

        for ev in events:
            # print(len(ev))
            segs = ev.get("segs", [])
            # print(len(segs))
            for s in segs:
                text = s.get("utf8", "").strip()
                if text:
                    lines.append(text)

        cleaned_text = " ".join(lines)

        return cleaned_text

    def get_transcription(self, url: str):
        """
        1. Tries fetching YouTube subtitles for transcription.
        2. If unavailable, downloads audio and transcribes.
        """
        try:
            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,
                'no_warnings': True,
                "subtitlesformat": "json3",
                "subtitleslangs": ["en"],
                "quiet": True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                subs = info.get("subtitles") or info.get("automatic_captions")

                if subs and "en" in subs:
                    transcript_url = subs["en"][0]["url"]

                    import requests
                    json_data = requests.get(transcript_url).text
                    new_data = json.loads(json_data)
                    with open(json_transcripts_path, "w") as jf:
                        json.dump(new_data, jf, indent=2)

                    from src.utils.yt_parser import parse_youtube_json_transcript
                    cleaned = parse_youtube_json_transcript(new_data)

                    print("[Ingestion] Found YouTube subtitles. Using them.")
                    return cleaned
                
            print("[Ingestion] No English subtitles found. Falling back to audio download.")
        
        except Exception as e:
            print(f"[Warning] Could not fetch subtitles: {e}")

        try:
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
                ydl.download([url])

            print(f"[Ingestion] Audio saved sucesssfully.")
        
        except Exception as e:
            raise RuntimeError(f"Audio download failed: {e}")

        print("[Transcription] Starting faster-whisper transcription...")
        model = WhisperModel("small", device="cpu", compute_type="int8")

        # Always make sure to get the transcripts in English
        segments, _ = model.transcribe(
            output+".mp3",
            task="translate",
            language="en")
        
        text = "\n".join([s.text for s in segments])
        with open("data/transcriptions.txt", "w") as f:
            f.write(text)
        print(f"[Transcription] Sucessfully transcribed using faster-whisper.")

        return text
            ...