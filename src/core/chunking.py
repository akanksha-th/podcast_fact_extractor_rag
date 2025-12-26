from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import nltk, os

cache_dir = os.path.join("data/", 'my_nltk_cache')
nltk.download('punkt', download_dir=cache_dir)


# ---------
def naive_chunking(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# ---------
def langchain_splitter(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n", 
            "\n", 
            ". ", 
            "! ", 
            "? ", 
            "; ", 
            ", ", 
            " ", 
            ""],
            keep_separator=True,
            length_function=len
    )
    chunks = splitter.split_text(text)

    cleaned_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 100:
            continue
        chunk = ensure_complete_sentences(chunk)
        cleaned_chunks.append(chunk)

    return cleaned_chunks

def ensure_complete_sentences(chunk: str) -> str:
    text = chunk.strip()
    if text and text[-1] not in {'.', '!', '?', '"', "'"}:
        last_period = text.rfind('.')
        last_exclaim = text.rfind('!')
        last_question = text.rfind('?')
        
        last_boundary = max(last_period, last_exclaim, last_question)
        
        if last_boundary > 0:
            # Cut off incomplete sentence
            text = text[:last_boundary + 1].strip()
    
    return text


# ---------
def semantic_chunks(text, max_chars=800, overlap=200):
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) <= max_chars:
            current += " " + sent
        else:
            if current.strip():
                chunks.append(current.strip())
            overlap_text = current[-overlap:] if overlap < len(current) else current
            current = overlap_text + " " + sent

    if current.strip():
        chunks.append(current.strip())

    return chunks