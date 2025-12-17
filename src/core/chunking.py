from langchain_text_splitters import RecursiveCharacterTextSplitter
import nltk, os
cache_dir = os.path.join("data/", 'my_nltk_cache')
nltk.download('punkt', download_dir=cache_dir)


# ---------
def naive_chunking(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# ---------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ".", " ", ""]
)
def langchain_splitter(text):
    return splitter.split_text(text)


# ---------
def semantic_chunks(text, max_chars=800, overlap=120):
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