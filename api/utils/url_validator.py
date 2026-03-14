import re
from urllib.parse import urlparse, parse_qs

class YouTubeService:
    """Check if the YouTube URL entered by user is correct or not"""
    def __init__(self):
        self.VALID_YT_HOSTS = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}
        self.VALID_ID_RE = re.compile(r'^[a-zA-Z0-9_-]{11}$')

    def extract_video_id(self, video_url: str) -> bool:
        if not video_url or not isinstance(video_url, str):
            return None
        
        try:
            parsed = urlparse(video_url)
        except Exception as e:
            return None
        
        if parsed.scheme not in ("http", "https"):
            return None
        if parsed.netloc not in self.VALID_YT_HOSTS:
            return None
        
        video_id = None
        if parsed.netloc == "youtu.be":
            video_id = parsed.path.strip("/").split("/")[0]
        elif parsed.path.startswith("/embed/"):
            video_id = parsed.path.split("/embed/")[-1].split("/")[0].split("?")[0]
        elif parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/shorts/")[-1].split("/")[0].split("?")[0]
        elif parsed.path.startswith("/watch"):
            params = parse_qs(parsed.query)
            ids = params.get("v", [])
            video_id = ids[0] if ids else None

        return video_id if video_id and self.VALID_ID_RE.match(video_id) else None
