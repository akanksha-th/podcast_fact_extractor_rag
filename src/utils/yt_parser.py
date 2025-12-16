import json

def parse_youtube_json_transcript(data: dict) -> str:
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

if __name__ == "__main__":
    json_path = "data/yt_json_transcripts.json"
    with open(json_path, "r") as jf:
        text = json.load(jf)

    clean = parse_youtube_json_transcript(text)
    print(clean[:500])