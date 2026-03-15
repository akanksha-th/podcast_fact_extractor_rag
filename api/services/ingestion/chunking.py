from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,        # number of characters -> 256 input tokens in embedder = ~1000 characters
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""]  # order of priority for splitting
        )

    def chunk_transcripts(self, text: str) -> list[str]:
        return self.text_splitter.split_text(text)