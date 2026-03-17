def build_rag_prompt(context: list[str], history: list[dict], question: str) -> str:
    context_text = "\n\n".join(context)
    history_text = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history])
    return f"""
You are a podcast analysis assistant.
Provided the context and history, answer the question asked by the user.
Answer ONLY from the transcript context provided. 
If the answer is not in the context, say "I couldn't find that in the podcast."

CONTEXT: {context_text}
HISTORY: {history_text}
QUESTION: {question}
"""

def build_map_prompt(context: str):
    return f"""You are a notes taker.
Given the context below, take important notes from them.
Format the notes into clear key points.
Notes should be between 100-200 words.
Do not skip any topic.

CONTEXT: {context}
"""
    
def build_reduce_prompt(context: str):
    return f"""You are a podcast notes generator.
Given the transcript context below, generate structured, detailed notes.
Format the notes with clear sections and key points.
Notes should be between 500-5000 words.
Focus on main topics, key insights, and important quotes.

CONTEXT: {context}
"""