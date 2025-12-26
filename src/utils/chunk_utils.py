import re
from itertools import islice
from typing import List

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


def clean_llm_bullets(text: str) -> List[str]:
    bullets = []

    for line in text.split("\n"):
        line = line.strip()

        line = re.sub(r'^[•\-\*\d+\)\.]\s*', '', line).strip()

        if not line or len(line) < 10:
            continue

        if line[-1] not in {'.', '!', '?'}:
            continue

        last_word = line.split()[-1].rstrip('.,!?').lower()
        if last_word in {'and', 'or', 'but', 'because', 'so', 'then'}:
            continue

        if len(line.split()) > 50:
            continue

        bullets.append(line)

    return bullets


if __name__ == "__main__":
    test_output = """
    1) Python was created in 1991
    2) It is
    3) The language emphasizes readability.
    4) Python supports multiple paradigms and
    5) It has a large standard library.
    """

    cleaned = clean_llm_bullets(test_output)
    print("Cleaned bullets:")
    for bullet in cleaned:
        print(f"{bullet}")