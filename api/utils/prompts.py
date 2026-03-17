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