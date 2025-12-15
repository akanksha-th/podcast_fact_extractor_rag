import re

def clean_transcript(text: str) -> str:
    # remove speaker labels
    text = re.sub(r"(Human:|System:)", "", text)

    # remove task instructions
    text = re.sub(r"##Your task:.*?Answer:", "", text, flags=re.DOTALL)

    # remove fillers
    text = re.sub(r"\[music\]", "", text)

    return text.strip()