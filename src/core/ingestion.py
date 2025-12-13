import yt_dlp, json
from faster_whisper import WhisperModel

json_transcripts_path = "data/yt_json_transcripts.json"

def get_transcription(url: str, output="data/podcast_01"):
    """
    1. Tries fetching YouTube subtitles for transcription.
    2. If unavailable, downloads audio and transcribes.
    """
    try:
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
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
                # with open(json_transcripts_path, "w") as jf:
                #     json.dump(new_data, jf, indent=2)

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
    model = WhisperModel("small")
    segments, _ = model.transcribe(output+".mp3")
    text = " ".join([s.text for s in segments])
    print(f"[Transcription] Sucessfully transcribed using faster-whisper.")

    return text

if __name__ == "__main__":
    url = "https://youtu.be/7ARBJQn6QkM?si=Ot5WxMcseHI-jPid"
    transcript = get_transcription(url)
    print(f"Length of transcript: {len(transcript)}")
    print(transcript[:500], "...")