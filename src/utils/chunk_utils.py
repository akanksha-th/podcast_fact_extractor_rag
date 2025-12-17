import re
from itertools import islice

def clean_transcript(text: str) -> str:
    text = re.sub(r"(Human:|System:)", "", text)
    text = re.sub(r"##Your task:.*?Answer:", "", text, flags=re.DOTALL)
    text = re.sub(r"\[music\]", "", text)

    return text.strip()


def batched(iterable, size):
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch


MAX_SECTION_CHARS = 1500
def chunk_sections(section_notes):
    chunks = []
    current = ""

    for note in section_notes:
        if len(current) + len(note) < MAX_SECTION_CHARS:
            current += "\n" + note
        else:
            chunks.append(current)
            current = note
    
    if current:
        chunks.append(current)

    return chunks