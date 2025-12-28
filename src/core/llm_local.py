from langchain_community.llms.gpt4all import GPT4All
from langchain_core.prompts import ChatPromptTemplate

llm = GPT4All(
    model="./models/phi-2.Q4_0.gguf",
    n_predict=128,  # lowered generation length from 256 to 128
    temp=0.1,
    top_k=40,
    top_p=0.9,
    repeat_penalty=1.1,
    verbose=False,
)

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


chunk_llm = GPT4All(
    model="./models/phi-2.Q4_0.gguf",
    n_predict=64,
    temp=0.0,
)

chunk_notes_prompt = ChatPromptTemplate([
    (
        "system",
        """You extract key facts from podcast transcripts.

        RULES:
        1. Write 3-5 bullet points
        2. Each bullet is ONE complete sentence
        3. Keep bullets under 15 words
        4. Use simple language
        5. Focus on main ideas only
        6. NO incomplete sentences
        7. NO trailing "and", "or", "but"

        EXAMPLES:

        Good bullets:
        • Python was created by Guido van Rossum in 1991.
        • The speaker recommends practicing daily for best results.
        • Machine learning models need large amounts of training data.

        Bad bullets (DON'T DO THIS):
        • Python was created by ← INCOMPLETE
        • The speaker recommends practicing daily and ← INCOMPLETE
        • Machine learning models need data because it ← INCOMPLETE

        Extract ONLY complete, standalone facts."""
            ),
    (
        "human",
        "Transcript:\n{trans_context}\n\nFacts:"
    )
])


section_llm = GPT4All(
    model="./models/phi-2.Q4_0.gguf",
    n_predict=96,
    temp=0.0,
)
sec_notes_prompt = ChatPromptTemplate([
    (
        "system",
        """You merge duplicate bullet points while keeping all unique information.

        TASK: Remove EXACT duplicates. Keep everything else.

        RULES:
        1. If two bullets say the SAME thing → keep one
        2. If two bullets say DIFFERENT things → keep both
        3. Each bullet MUST be a complete sentence
        4. NO incomplete bullets
        5. Preserve bullet format (•)

        EXAMPLES:

        Input:
        • Python is fast
        • Python is fast
        • Python is efficient

        Output:
        • Python is fast
        • Python is efficient

        Input:
        • The speaker discussed AI
        • AI was mentioned by the speaker
        • Machine learning requires data

        Output:
        • The speaker discussed AI
        • Machine learning requires data"""
            ),
    (
        "human",
        "Bullets to merge:\n{chunk_notes}\n\nMerged bullets:"
    )
])