from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
import os

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

if not HF_API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    except:
        pass


llm = HuggingFaceEndpoint(
    repo_id="microsoft/phi-2",
    temperature=0.1,
    max_new_tokens=128,
    huggingfacehub_api_token=HF_API_KEY,
    timeout=120
    )

# RAG Prompt
rag_prompt = ChatPromptTemplate([
    (
        "system",
        "You answer questions ONLY about the podcast transcript. "
        "The speaker refers to the human speaker in the podcast, NOT the AI. "
        "If the answer is not in the context, say 'I don't know'. "
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])

# Chunk notes LLM
chunk_llm = HuggingFaceEndpoint(
    repo_id="microsoft/phi-2",
    temperature=0.0,
    max_new_tokens=64,
    huggingfacehub_api_token=HF_API_KEY,
    timeout=120
)

# Chunk notes prompt
chunk_notes_prompt = ChatPromptTemplate([
    (
        "system",
        """Extract 3-5 key facts. Each fact must be:
- A complete sentence
- Under 15 words
- End with punctuation

Example:
- Python was created by Guido van Rossum in 1991.
- The language emphasizes code readability.
- Python is used for web development and data science."""
    ),
    (
        "human",
        "Transcript:\n{trans_context}\n\nFacts:"
    )
])

# Section notes LLM
section_llm = HuggingFaceEndpoint(
    repo_id="microsoft/phi-2",
    temperature=0.0,
    max_new_tokens=96,
    huggingfacehub_api_token=HF_API_KEY,
    timeout=120
)

# Section notes prompt
sec_notes_prompt = ChatPromptTemplate([
    (
        "system",
        """Remove duplicate bullet points. Keep all unique information.

Rules:
- Each bullet must be complete sentence
- Remove only exact duplicates
- Keep all unique points"""
    ),
    (
        "human",
        "Bullets:\n{chunk_notes}\n\nMerged:"
    )
])