from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Optional, Dict
from IPython.display import display, Image
from datetime import datetime
import os

from src.core.ingestion import get_transcription
from src.core.chunking import langchain_splitter
from sentence_transformers import SentenceTransformer
from src.core.storage import store_vectors, fetch_emb
from src.utils.chunk_utils import clean_transcript, batched, clean_llm_bullets
from src.core.llm import (
    llm, rag_prompt,
    chunk_llm, section_llm,
    chunk_notes_prompt, sec_notes_prompt, 
)
from config import settings
from src.utils.logger import get_logger, LoggerContext

logger = get_logger(__name__)


class ExtractorState(TypedDict):
    url: str
    transcripts: str
    chunks: List[str]
    embeddings: List
    query: str      # "text" | "audio"
    retrieved_docs: List[str]
    answer: str
    should_exit: bool
    task: str
    notes: str
    error: Optional[str]
    metadata: Dict


class ModelManager:
    """Manages model loading and caching"""
    _embedder = None

    @classmethod
    def get_embedder(cls):
        if cls._embedder is None:
            with LoggerContext(logger, f"Loading embedding model: {settings.embedding_model}"):
                try:
                    cls._embedder = SentenceTransformer(settings.embedding_model)
                    logger.info(f"Embedder Loaded: {settings.embedding_model}")
                except Exception as e:
                    logger.error(f"Failed to load embedder: {e}")
                    raise
        return cls._embedder
    

def transcription_node(state: ExtractorState) -> ExtractorState:
    """Extracts video transcripts"""
    with LoggerContext(logger, "transcription"):
        try:
            url = state["url"].strip()
            if not url:
                raise ValueError("URL is empty.")
            
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Invalid URL format: {url}")
            
            logger.info(f"Fetching transcript from: {url}")
            transcripts = get_transcription(url)

            if not transcripts or len(transcripts.strip()) < settings.min_transcript_length:
                raise ValueError(
                    f"Transcript too short or empty (got {len(transcripts)} chars, "
                    f"minimum {settings.min_transcript_length})"
                )

            state["error"] = None
            state["transcripts"] = transcripts
            state["metadata"]["transcript_length"] = len(transcripts)

            logger.info(f"Transcription successful, length: {len(transcripts)} characters.")
            print("\n[Agent] Transcription completed.\n")

        except Exception as e:
            error_msg = f"Transcription failed: {e}"
            logger.error(error_msg, exc_info=True)

            state["error"] = error_msg
            state["transcripts"] = ""

            print(f"\n[Agent] {error_msg}\n")
            print("\n  Please check the URL and try again.")
    
    return state


def chunk_and_embed_node(state: ExtractorState) -> ExtractorState:
    """Chunks the transcripts and embeds them"""
    with LoggerContext(logger, "chunking and embedding"):
        try:
            if state.get("error"):
                logger.warning("Skipping due to previous errors.")
                return state
            
            text = state["transcripts"]
            if not text:
                raise ValueError("No transcripts available for chunking.")
            
            logger.info("Starting chunking...")
            chunks = langchain_splitter(text)

            if not chunks:
                raise ValueError("Chunking resulted in no chunks.")
            
            chunks = [c for c in chunks if len(c.strip()) > settings.min_chunk_size]
            logger.info(f"Chunking completed: {len(chunks)} chunks created.")

            embedder = ModelManager.get_embedder()
            logger.info("Generating embedding...")
            embeddings = embedder.encode(
                chunks,
                batch_size=settings.embedding_batch_size,
                show_progress_bar=True
            )

            state["chunks"] = chunks
            state["embeddings"] = embeddings
            state["error"] = None
            state["metadata"]["num_chunks"] = len(chunks)
            state["metadata"]["embedding_dim"] = embeddings.shape[1]

            logger.info(f"Created {len(chunks)} chunks and embeddings")
            print("\n[Agent] Chunking and embedding completed.\n")
        
        except Exception as e:
            error_msg = f"Chunking/Embedding failed: {e}"
            logger.error(error_msg, exc_info=True)

            state["error"] = error_msg
            state["chunks"] = []
            state["embeddings"] = []

            print(f"\n[Agent] {error_msg}\n")

    return state


def store_node(state: ExtractorState) -> ExtractorState:
    """Stores vectors in Qdrant"""
    with LoggerContext(logger, "storing vectors"):
        try:
            if state.get("error"):
                logger.warning("Skipping due to previous error")
                return state
            
            if not state["chunks"] and len(state["embeddings"]) == 0:
                raise ValueError("No chunks or embeddings to store")
            
            logger.info(f"Storing {len(state['chunks'])} vectors in Qdrant...")

            store_vectors(
                name=settings.qdrant_collection,
                chunks=state["chunks"],
                embeddings=state["embeddings"]
            )

            state["errors"] = None
            logger.info(f"Stored {len(state['chunks'])} vectors")
            print(f"\n[Agent] Vectors stored in collection: {settings.qdrant_collection}\n")

        except Exception as e:
            error_msg = f"Storing Failed: {e}"
            logger.error(error_msg, exc_info=True)

            state["error"] = error_msg
            print("\n[Agent] {error_msg}\n")
            print("\n  Check if Qdrant is running")

    return state

# -----
def choose_task_node(state: ExtractorState) -> ExtractorState:
    try:
        if state.get("error"):
            print("\n[Agent] Cannot continue due to previous error\n")
            state["task"] = "exit"
            return state
        
        print("\n" + "-"*60)
        print("Choose Task:\n  1. Ask questions\n  2. Generate Notes")
        print("-"*60)

        while True:
            task = input("Enter your choice (1 or 2): ").strip()

            if task in settings.valid_tasks:
                state["task"] = task
                state["error"] = None
                logger.info(f"Task selected: {'Q&A' if task == '1' else 'Notes'}")
                break
            else:
                print("Invalid Choice. Please enter 1 or 2")

    except KeyboardInterrupt:
        print("\n[Agent] Interrupted by user. Exiting...\n")
        state["task"] = "exit"

    except Exception as e:
        logger.error(f"Task selection failed: {e}", exc_info=True)
        state["error"] = str(e)
        state["task"] = "exit"
    
    return state

def route_task(state: ExtractorState) -> str:
    if state["task"] == "exit" or state.get("error"):
        return "end"
    return "get_query" if state["task"] == "1" else "make_notes"


# -----
def get_query_node(state: ExtractorState) -> ExtractorState:
    try:
        query = input("\nAsk a question (type 'exit' to quit)").strip()
        
        if not query:
            print("Empty query. Please try again.")
            state["query"] = ""
            state["should_exit"] = False
            return state
        
        if len(query) > settings.max_query_length:
            print(f" Query too long (max {settings.max_query_length} chars). Truncating...")
            query = query[:settings.max_query_length]

        state["query"] = query
        state["should_exit"] = True if query.lower() in {"exit", "quit"} else False
        state["error"] = None
        logger.info(f"Query received: {query[:50]}")

    except KeyboardInterrupt:
        print("\n[Agent] Interrupted by user. Exiting...\n")
        state["should_exit"] = True

    except Exception as e:
        logger.error(f"Task selection failed: {e}", exc_info=True)
        state["error"] = str(e)
        state["should_exit"] = True

    return state

def retrieve_node(state: ExtractorState) -> ExtractorState:
    """Retrieves the embeddings"""
    with LoggerContext(logger, "retreiving relevant chunks"):
        try:
            query = state["query"]
            embedder = ModelManager.get_embedder()

            q_emb = embedder.encode(query)

            candidates = fetch_emb(
                name=settings.qdrant_collection,
                query_emb=q_emb,
                limit=settings.top_k_retrieval * 2
            )
            docs = candidates[:settings.top_k_retrieval]

            if not docs:
                logger.warning("No relevant documents found")
                print("No relevant information found for your query")
                state["retrieved_docs"] = []

            else:
                total_length = 0
                filtered_docs = []
                for doc in docs:
                    if total_length + len(doc) <= settings.max_context_length:
                        filtered_docs.append(doc)
                        total_length += len(doc)
                    else:
                        break
                
                state["retrieved_docs"] = filtered_docs
                logger.info(f"Retrieved {len(filtered_docs)} relevant chunks")
            
            state["error"] = None

        except Exception as e:
            error_msg = f"Retreival Failed: {str(e)}"
            logger.error(error_msg, exc_info=True)

            state["error"] = error_msg
            state["retrieved_docs"] = []
            print(f"\n[Agent] {error_msg}\n")
    
    return state

def generate_node(state: ExtractorState) -> ExtractorState:
    """generates answers to the user query"""
    with LoggerContext(logger, "generating answer"):
        try:
            if not state["retrieved_docs"]:
                answer = "I don't have enough information to answer the question, based on this transcript"
                state["answer"] = answer
                print(f"\n[Agent] Answer: {answer}\n")
                return state
            
            context = "\n\n".join(state["retrieved_docs"])
            query = state["query"]
            logger.info(f"Generating Answer: {len(context)} chars...")

            rag_chain = rag_prompt | llm
            max_attempts = 3

            for attempt in range(max_attempts):
                try:
                    answer = rag_chain.invoke({
                        "context": context,
                        "question": query
                    })
                    break
                except Exception as e:
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                        continue
                    else:
                        raise
            
            state["answer"] = answer
            state["error"] = None

            logger.info(f"Answer generated: {len(answer)} chars")
            print(f"\n[Agent] Answer: {answer}\n")

        except Exception as e:
            error_msg = f"Generation Failed: {str(e)}"
            logger.error(error_msg, exc_info=True)

            state["error"] = error_msg
            state["answer"] = "Sorry, I couldn't generate an answer. Please try again."
            print(f"\n[Agent] {error_msg}\n")

    return state

# -----
def notes_node(state: ExtractorState) -> ExtractorState:
    """Generate hierarchical notes with progress tracking"""
    with LoggerContext(logger, "notes generation"):
        try:
            if state.get("error"):
                print("\n[Agent] Cannot generate notes due to previous error\n")
                return state
            
            print("\n[Agent] Starting notes generation...\n")
            print("\t\tThis may take a few minutes...\n")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chunk_path, section_path = settings.get_notes_path(timestamp)

            # ------ Stage 1: Chunk Notes -----
            chunk_notes = []
            chunk_notes_chain = chunk_notes_prompt | chunk_llm

            logger.info("Stage 1: Chunk Notes generation...")

            for i, chunk in enumerate(state["chunks"]):
                chunk = clean_transcript(chunk)
                if len(chunk.strip()) < settings.min_chunk_size:
                    continue

                try:
                    raw_notes = chunk_notes_chain.invoke({"trans_context": chunk})
                    clean_bullets = clean_llm_bullets(raw_notes)

                    if clean_bullets:
                        formatted_notes = "\n".join([f"• {b}" for b in clean_bullets])
                        chunk_notes.append(formatted_notes)

                    if settings.save_intermediate_notes:
                        open("outputs/chunk_notes.md", "a", encoding="utf-8").write(formatted_notes + "\n")

                    if i % 10 == 0:
                        print(f"\t\t Progress: {i+1}/{len(state['chunks'])}")

                except Exception as e:
                    logger.warning(f"Chunk {i} failed: {e}")
                    continue

            logger.info(f"Generated {len(chunk_notes)} chunk notes")
            print("\n[Agent] Generated {len(chunk_notes)} chunk notes\n")

            # ------ Stage 2: Section Notes -----
            section_notes = []
            sec_notes_chain = sec_notes_prompt | section_llm

            logger.info("Stage 2: Section Notes generation...")

            sections = list(batched(chunk_notes, size=settings.section_batch_size))
            for i, batch in enumerate(sections):
                try:
                    raw_sec = sec_notes_chain.invoke({"chunk_notes":"\n".join(batch)})
                    clean_bullets = clean_llm_bullets(raw_sec)
                    if clean_bullets:
                        formatted_sec = "\n".join([f"• {b}" for b in clean_bullets])
                        section_notes.append(formatted_sec)

                    if settings.save_intermediate_notes:
                        open("outputs/section_notes.md", "a", encoding="utf-8").write(formatted_sec + "\n\n")

                    if i % 10 == 0:
                        print(f"[Notes] Processed sections {i+1}/{int(len(state['chunks'])/6)}")

                except Exception as e:
                    logger.warning(f"Section {i} failed: {e}")
                    continue

            final_notes = "\n\n".join(section_notes)
            state["notes"] = final_notes
            state["error"] = None

            logger.info("Notes generation completed")
            print("\n[Agent] Notes generation completed.\n")

            print(f"Notes: {state['notes']}")

        except Exception as e:
            error_msg = f"Notes Generation Failed: {str(e)}"
            logger.error(error_msg, exc_info=True)

            state["error"] = error_msg
            state["notes"] = []
            print(f"\n[Agent] {error_msg}\n")

    return state

#-----
def should_continue(state: ExtractorState) -> str:
    """Determine if we should continue the loop"""
    if state["should_exit"] or state.get("error"):
        return "end"
    return "retrieve"


def build_graph() -> StateGraph:
    logger.info("Building agent graph...")

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
            "make_notes": "make_notes",
            "end": END
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

    logger.info("Graph built successfully")

    return graph.compile()


if __name__ == "__main__":
    # run on CLI using "python -m src.agents.agent_v2"

    print("="*60)
    print("Podcast RAG Agent v2")
    print("="*60)

    if not settings.validate_model_exists():
        logger.error(f"LLM model not found: {settings.llm_model_path}")
        print(f"\nError: Model file not found")
        print(f"Expected location: {settings.llm_model_path}")
        print(f"Please download the model first.")
        exit(1)
    
    # Example URLs
    url = input("\nEnter YouTube URL: ").strip()
    if not url:
        url = "https://youtu.be/UPk56BR1Cmk?si=sRPakuv_DlAkbYfG"
        print(f"   Using default: {url}")
    
    # Build and run graph
    logger.info("Starting agent execution...")
    app = build_graph()
    
    try:
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
            "notes": "",
            "error": None,
            "metadata": {}
        })
        
        logger.info("Agent execution completed")
        print("\n" + "="*60)
        print("Session completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n[Agent] Interrupted by user. Exiting...\n")
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        print(f"\nFatal error: {e}")
        print("   Check logs for details")


    img_bytes = app.get_graph().draw_mermaid_png()
    os.makedirs("graph_images", exist_ok=True)
    output_path = "graph_images/agent_graph_v2.png"

    with open(output_path, "wb") as f:
        f.write(img_bytes)
    print(f"Graph saved to: {output_path}")