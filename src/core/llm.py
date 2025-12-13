from langchain_community.llms.gpt4all import GPT4All
from langchain_core.prompts import ChatPromptTemplate

llm = GPT4All(
    model="./models/phi-2.Q4_0.gguf",
    n_predict=256,
    temp=0.2,
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

