from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from src.core.ingestion import get_transcription
from sentence_transformers import SentenceTransformer
from src.core.storage import store_vectors, fetch_emb
from src.utils.cleaner import clean_transcript
from src.core.llm import llm, rag_prompt, notes_prompt
from IPython.display import display, Image
import os


class ExtractorState(TypedDict):
    url: str
    transcripts: str
    chunks: List[str]
    embeddings: List
    query: str
    retrieved_docs: List[str]
    answer: str
    should_exit: bool
    task: str
    notes: str


def transcription_node(state: ExtractorState) -> ExtractorState:
    """Extracts video transcripts"""
    transcripts = get_transcription(state["url"])
    state["transcripts"] = transcripts
    print("[Agent] Transcription completed.")
    return state

embedder = SentenceTransformer("intfloat/multilingual-e5-base")
def chunk_and_embed_node(state: ExtractorState) -> ExtractorState:
    """Chunks the transcripts and embeds them"""
    chunk_size = 500
    text = state["transcripts"]
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    embeddings = embedder.encode(chunks)

    state["chunks"] = chunks
    state["embeddings"] = embeddings

    print(f"[Agent] Created {len(chunks)} chunks and embeddings.")
    return state

def store_node(state: ExtractorState) -> ExtractorState:
    """Stores vectors in Qdrant"""
    store_vectors(
        name="podcast_01",
        chunks=state["chunks"],
        embeddings=state["embeddings"]
    )
    print("[Agent] Stored vectors in Qdrant.")
    return state

# -----
def choose_task_node(state: ExtractorState) -> ExtractorState:
    print("Choose Task:\n  1. Ask questions\n  2. Generate Notes")
    state["task"] = input("Enter your choice (1 or 2): ")
    return state

def route_task(state: ExtractorState) -> str:
    return "get_query" if state["task"] == "1" else "make_notes"

# -----
def notes_node(state: ExtractorState) -> ExtractorState:
    notes_chain = notes_prompt | llm
    notes_per_chunk = []

    for chunk in state["chunks"][:20]:  # Chunk cap
        chunk = clean_transcript(chunk)
        chunk_notes = notes_chain.invoke({
            "trans_context": chunk
        })
        notes_per_chunk.append(chunk_notes)

    state["notes"] = "\n\n".join(notes_per_chunk)
    print("[Agent] Notes generated.")

    with open("outputs/podcast_notes.md", "w", encoding="utf-8") as f:
        f.write(state["notes"])

    print(f"Notes: {state['notes']}")

    return state

# -----
def get_query_node(state: ExtractorState) -> ExtractorState:
    query = input("\nAsk a question (type 'exit' to quit)").strip()
    state["query"] = query
    state["should_exit"] = True if query.lower() in {"exit", "quit"} else False

    return state

def retrieve_node(state: ExtractorState) -> ExtractorState:
    """Retrieves the embeddings"""
    q_emb = embedder.encode(state["query"])
    docs = fetch_emb(name="podcast_01", query_emb=q_emb, limit=5)

    state["retrieved_docs"] = docs
    # print("[Agent] Retrieved relevant chunks.")
    return state

def generate_node(state: ExtractorState) -> ExtractorState:
    """generates answers to the user query"""
    context = "\n".join(state["retrieved_docs"])
    query = state["query"]

    # COMPOSITION
    rag_chain = rag_prompt | llm
    answer = rag_chain.invoke({
        "context": context,
        "question": query
    })

    state["answer"] = answer
    print("\n[Agent] Answer:\n", answer)
    # print("[Agent] Answer generated")

    return state

#-----
def should_continue(state: ExtractorState) -> str:
    return "end" if state["should_exit"] else "retrieve"


def build_graph() -> StateGraph:
    graph = StateGraph(ExtractorState)

    graph.add_node("transcription", transcription_node)
    graph.add_node("embed", chunk_and_embed_node)
    graph.add_node("store", store_node)
    graph.add_node("choose_task", choose_task_node)
    graph.add_node("make_notes", notes_node)
    graph.add_node("get_query", get_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "transcription")
    graph.add_edge("transcription", "embed")
    graph.add_edge("embed", "store")
    graph.add_edge("store", "choose_task")

    graph.add_conditional_edges(
        "choose_task",
        route_task,
        {
            "get_query": "get_query",
            "make_notes": "make_notes"
        }
    )
    graph.add_edge("make_notes", END)

    graph.add_conditional_edges(
        "get_query",
        should_continue,
        {
            "retrieve": "retrieve",
            "end": END
        }
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "get_query")

    return graph.compile()


if __name__ == "__main__":
    # run on CLI using "python -m src.agents.agent_v1"

    url = "https://youtu.be/cUbe6HbFncE?si=ellaHl0A8pERo_OH"
    app = build_graph()

    result = app.invoke({
        "url": url,
        "transcripts": "",
        "chunks": [],
        "embeddings": [], 
        "query": "", 
        "retrieved_docs": [],
        "answer": "",
        "should_exit": False,
        "task": "",
        "notes": ""
    })

    img_bytes = app.get_graph().draw_mermaid_png()
    os.makedirs("graph_images", exist_ok=True)
    output_path = "graph_images/agent_graph_v1.png"

    with open(output_path, "wb") as f:
        f.write(img_bytes)
    print(f"Graph saved to: {output_path}")