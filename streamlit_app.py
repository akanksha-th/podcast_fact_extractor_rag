import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))
from src.core.ingestion import get_transcription
from src.core.chunking import langchain_splitter
from sentence_transformers import SentenceTransformer
from src.core.storage import store_vectors, fetch_emb
from src.utils.chunk_utils import clean_transcript, batched
from src.core.llm import (
    llm, rag_prompt,
    chunk_llm, section_llm,
    chunk_notes_prompt, sec_notes_prompt
)
from src.utils.logger import get_logger
from config import settings
import time
from datetime import datetime

logger = get_logger(__name__)

st.set_page_config(
    page_title="Podcast RAG Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.write("You can ask questions about the podcast episode and get quick notes")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if "processed" not in st.session_state:
    st.session_state.processed = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


@st.cache_resource
def load_embedder():
    with st.spinner("Loading embedding model..."):
        model = SentenceTransformer(settings.embedding_model)
    return model

def process_podcast(url):

    results = {
        "success": False,
        "error": None,
        "transcript_length": 0,
        "num_chunks": 0,
        "processing_time": 0
    }

    start = time.time()

    try:
        with st.spinner("Processing podcast episode..."):
            progress_bar = st.progress(0)
            transcript = get_transcription(url)
            progress_bar.progress(37)

            if not transcript or len(transcript.strip()) == 0:
                results["error"] = "Failed to retrieve transcript."
                raise ValueError("Failed to retrieve transcript.")
            
            results["transcript_length"] = len(transcript)
            st.success("Transcript fetched successfully!")

        with st.spinner("Creating Chunks and Embeddings..."):
            chunks = langchain_splitter(transcript)
            progress_bar.progress(63)

            embedder = load_embedder()
            embeddings = embedder.encode(chunks, show_progress_bar=False, batch_size=32)
            progress_bar.progress(100)

            results["num_chunks"] = len(chunks)
            st.success("Chunks and embeddings created successfully!")

        with st.spinner("Storing embeddings..."):
            store_vectors(
                name=settings.qdrant_collection,
                chunks=chunks,
                embeddings=embeddings
            )
            st.success("Embeddings stored in Qdrant!")

        st.session_state.processed = True
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings

        results["success"] = True
        results["processing_time"] = time.time() - start

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"Error processing podcast: {e}", exc_info=True)
    
    return results


def answer_questions(question: str) -> str:
    start = time.time()

    try:
        embedder = load_embedder()
        q_emb = embedder.encode(question)
        docs = fetch_emb(
            name=settings.qdrant_collection,
            query_emb=q_emb,
            limit=settings.top_k_retrieval
        )

        if not docs:
            return {
                "answer": "I could not find relevant informaton to answer this question",
                "sources": [],
                "time": time.time() - start
            }
        
        context = "\n\n".join(docs)
        rag_chain = rag_prompt | llm
        answer = rag_chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "answer": answer,
            "sources": docs,
            "time": time.time() - start
        }
    
    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        return {
            "answer": f"ERROR: {str(e)}",
            "sources": [],
            "time": time.time() - start
        }


def generate_notes() -> str:
    if not st.session_state.chunks:
        return "No processed podcast. Enter a URL first."
    
    try:
        with st.spinner("Generating quick notes..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            chunk_notes = []
            chunk_notes_chain = chunk_notes_prompt | chunk_llm

            total_chunks = len(st.session_state.chunks)

            for i, chunk in enumerate(st.session_state.chunks):
                chunk_clean = clean_transcript(chunk)
                if len(chunk_clean.strip()) < 200:
                    continue
                
                try:
                    note = chunk_notes_chain.invoke({"trans_context": chunk_clean})
                    chunk_notes.append(note)
                except:
                    continue
                
                # Update progress
                progress = (i + 1) / total_chunks * 0.7
                progress_bar.progress(progress)
                status_text.text(f"Processing chunk {i+1}/{total_chunks}")

            section_notes = []
            sec_notes_chain = sec_notes_prompt | section_llm
            
            sections = list(batched(chunk_notes, size=4))
            
            for i, batch in enumerate(sections):
                try:
                    sec = sec_notes_chain.invoke({"chunk_notes": "\n".join(batch)})
                    section_notes.append(sec)
                except:
                    continue
                
                # Update progress
                progress = 0.7 + (i + 1) / len(sections) * 0.3
                progress_bar.progress(progress)
                status_text.text(f"Merging section {i+1}/{len(sections)}")
            
            progress_bar.progress(1.0)
            status_text.text("✓ Notes generated!")
            
            final_notes = "\n\n".join(section_notes)
            return final_notes
        
    except Exception as e:
        logger.error(f"Notes generation failed: {e}", exc_info=True)
        return f"Error generating notes: {str(e)}"
    

def main():
    with st.sidebar:
        st.header("Settings")

        st.info(f"""
        - **Embedding Model**: {settings.embedding_model}
        - **LLM Model**: {settings.llm_model}
        - **Chunk Size**: {settings.chunk_size}
        - **Top-K Retrievals**: {settings.top_k_retrieval}
        """)

        with st.expander("Example Podcasts"):
            st.markdown("""
            - [The Joe Rogan Experience - Elon Musk](https://www.youtube.com/watch?v=ycPr5-27vSI)
            - [Lex Fridman Podcast - Richard Dawkins](https://www.youtube.com/watch?v=8mve0f3WOtg)
            - [The Tim Ferriss Show - Naval Ravikant](https://www.youtube.com/watch?v=5h0J6z7kK1g)
            """)

        if st.session_state.processed:
            st.success("Podcast Processed")
            st.metric("Chunks", len(st.session_state.chunks))
            st.metric("Embeddings", len(st.session_state.embeddings))

    tab1, tab2, tab3 = st.tabs(["Load Podcast", "Ask Questions", "Get Quick Notes"])

    with tab1:
        st.header("Process a podcast")

        url = st.text_input(
            "Youtube URL",
            placeholder="https://youtube.com/watch?v=....",
            help="Enter a Youtube URL"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            process_btn = st.button("Process", type="primary", use_container_width=True)

        if process_btn and url:
            results = process_podcast(url)

            if results["success"]:
                st.balloons()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{results['transcript_length']:,}</h3>
                        <p>Characters</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{results['num_chunks']}</h3>
                        <p>Chunks</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{results['processing_time']:.1f}s</h3>
                        <p>Time</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success("Ready to answer questions!")
            else:
                st.error(f"Error: {results['error']}")
            
        elif process_btn:
            st.warning("Please enter a URL")

    with tab2:
        st.header("Ask Questions")
        if not st.session_state.processed:
            st.info("Please process a podcast first")

        else:
            question = st.text_input(
                "Your Question",
                placeholder="Enter your question about the podcast...",
                key="question_input"
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                ask_btn = st.button("Ask", type="primary", use_container_width=True)

            with col2:
                clear_btn = st.button("Clear History", use_container_width=True)

            if clear_btn:
                st.session_state.chat_history = []
                st.rerun()

            if ask_btn and question:
                response = answer_questions(question)

                st.session_state.chat_history.append({
                    'question': question,
                    'answer': response['answer'],
                    'time': response['time'],
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })

                if st.session_state.chat_history:
                    st.divider()
                    
                    for i, chat in enumerate(reversed(st.session_state.chat_history)):
                        with st.container():
                            st.markdown(f"**Question ({chat['timestamp']}):**")
                            st.info(chat['question'])
                            
                            st.markdown("**Answer:**")
                            st.success(chat['answer'])
                            
                            st.caption(f"Response time: {chat['time']:.2f}s")
                        st.divider()
    
    with tab3:
        st.header("Generate Quick Notes")
        if not st.session_state.processed:
            st.info("Please process a podcast first")
        
        else:
            notes_btn = st.button("Generate Notes", type="primary")

            if notes_btn:
                notes = generate_notes()
                st.markdown("Generated Notes")
                st.markdown(notes)
                
                # Download button
                st.download_button(
                    label="⬇️ Download Notes",
                    data=notes,
                    file_name=f"podcast_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )


if __name__ == "__main__":
    main()