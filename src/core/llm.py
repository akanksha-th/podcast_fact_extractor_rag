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
        "You extract factual notes from a podcast transcript.\n"
        "Summarize the transcript into 3–5 SHORT bullet points. "
        "Each bullet MUST be under 12 words. "
        "No explanations. No repetition."
    ),
    (
        "human",
        "Transcript:\n{trans_context}\n\n Bullet Notes:"
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
        "You are merging podcast notes.\n"
        "Rules:\n"
        "- Merge overlapping ideas\n"
        "- Remove repetition\n"
        "- Keep factual points only\n"
        "- Preserve the original order\n"
        "- Do NOT add new information\n"
    ),
    (
        "human",
        "Notes:\n{chunk_notes}\n\n Merged Notes:"
    )
])

notes_llm = GPT4All(
    model="./models/phi-2.Q4_0.gguf",
    n_predict=128,
    temp=0.0,
)
final_notes_prompt = ChatPromptTemplate([
    (
        "system",
        "You are organizing podcast notes into a clean structure.\n"
        "Rules:\n"
        "- Use clear headings\n"
        "- Group related points"
        "- Keep bullet points concise\n"
        "- Do NOT invent content\n"
        "- Do NOT repeat ideas\n"
    ),
    (
        "human",
        "Notes:\n{section_notes}\n\n Structured Notes:"
    )
])