import json

def parse_youtube_json_transcript(raw_text: str) -> str:
    """
    Converts Youtube json subtitle format into plain text.
    """
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text
    
    events = data.get("events", [])
    lines = []

    for ev in events:
        segs = ev.get("segs", [])
        for s in segs:
            text = s.get("utf-8", "").strip()
            if text:
                lines.append(text)

    return " ".join(lines)