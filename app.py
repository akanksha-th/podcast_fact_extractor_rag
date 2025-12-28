import gradio as gr
import os
from pathlib import Path

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
from config import settings

embedder = None
chunks = []
embeddings = []


def load_embedder():
    """Load embedder model (cached globally)"""
    global embedder
    if embedder is None:
        embedder = SentenceTransformer(settings.embedding_model)
    return embedder


def process_podcast(url, progress=gr.Progress()):
    """Process a podcast URL"""
    global chunks, embeddings
    
    if not url:
        return "Please enter a YouTube URL", "", ""
    
    try:
        progress(0.1, desc="Fetching transcript...")
        transcript = get_transcription(url)
        
        if not transcript or len(transcript) < 100:
            return "Transcript too short or empty", "", ""
        
        progress(0.4, desc="Creating chunks...")
        chunks = langchain_splitter(transcript)
        
        progress(0.6, desc="Generating embeddings...")
        emb = load_embedder()
        embeddings = emb.encode(chunks, show_progress_bar=False)
        
        progress(0.8, desc="Storing in vector database...")
        store_vectors(
            name=settings.qdrant_collection,
            chunks=chunks,
            embeddings=embeddings
        )
        
        progress(1.0, desc="Complete!")
        
        stats = f"""
### Processing Complete!

- **Transcript Length:** {len(transcript):,} characters
- **Chunks Created:** {len(chunks)}
- **Embeddings Generated:** {len(embeddings)}

You can now ask questions!
        """
        
        return stats, gr.update(visible=True), gr.update(visible=True)
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, gr.update(visible=False), gr.update(visible=False)


def answer_question(question, chat_history):
    """Answer a question using RAG"""
    
    if not chunks:
        return chat_history + [[question, "Please process a podcast first!"]]
    
    if not question or len(question.strip()) == 0:
        return chat_history + [[question, "Please enter a question."]]
    
    try:
        emb = load_embedder()
        q_emb = emb.encode(question)
        docs = fetch_emb(
            name=settings.qdrant_collection,
            query_emb=q_emb,
            limit=settings.top_k_retrieval
        )
        
        if not docs:
            answer = "I couldn't find relevant information to answer this question."
        else:
            context = "\n\n".join(docs)
            rag_chain = rag_prompt | llm
            answer = rag_chain.invoke({
                "context": context,
                "question": question
            })
        
        chat_history = chat_history + [[question, answer]]
        return chat_history
    
    except Exception as e:
        error_answer = f"Error: {str(e)}"
        return chat_history + [[question, error_answer]]


def generate_notes(progress=gr.Progress()):
    """Generate structured notes"""
    
    if not chunks:
        return "Please process a podcast first!"
    
    try:
        progress(0.1, desc="Generating chunk notes...")
        chunk_notes = []
        chunk_notes_chain = chunk_notes_prompt | chunk_llm
        
        for i, chunk in enumerate(chunks):
            progress(0.1 + (i / len(chunks) * 0.6), desc=f"Processing chunk {i+1}/{len(chunks)}")
            
            chunk_clean = clean_transcript(chunk)
            if len(chunk_clean.strip()) < 200:
                continue
            
            try:
                note = chunk_notes_chain.invoke({"trans_context": chunk_clean})
                chunk_notes.append(note)
            except:
                continue
        
        progress(0.7, desc="Merging sections...")
        section_notes = []
        sec_notes_chain = sec_notes_prompt | section_llm
        
        sections = list(batched(chunk_notes, size=4))
        
        for i, batch in enumerate(sections):
            progress(0.7 + (i / len(sections) * 0.3), desc=f"Merging section {i+1}/{len(sections)}")
            
            try:
                sec = sec_notes_chain.invoke({"chunk_notes": "\n".join(batch)})
                section_notes.append(sec)
            except:
                continue
        
        progress(1.0, desc="Notes generated!")
        
        # Format final notes
        final_notes = "# Podcast Notes\n\n"
        final_notes += "\n\n".join(section_notes)
        
        return final_notes
    
    except Exception as e:
        return f"Error generating notes: {str(e)}"


# ==========================================
# GRADIO INTERFACE
# ==========================================

# Custom CSS
css = """
.gradio-container {
    max-width: 1200px;
    margin: auto;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
"""

# Create interface
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.HTML("""
        <div class="header">
            <h1>Podcast RAG Assistant</h1>
            <p>AI-powered question answering for podcast transcripts</p>
        </div>
    """)
    
    # Tab 1: Process Podcast
    with gr.Tab("Process Podcast"):
        gr.Markdown("""
        ### Step 1: Enter a YouTube URL
        The podcast should have subtitles/captions available.
        """)
        
        with gr.Row():
            url_input = gr.Textbox(
                label="YouTube URL",
                placeholder="https://youtube.com/watch?v=...",
                scale=4
            )
            process_btn = gr.Button("Process", variant="primary", scale=1)
        
        status_output = gr.Markdown(label="Status")
        
        gr.Markdown("""
        ### Example URLs:
        - `https://youtu.be/UPk56BR1Cmk`
        - `https://youtu.be/cUbe6HbFncE`
        """)
    
    # Tab 2: Ask Questions
    with gr.Tab("Ask Questions") as qa_tab:
        gr.Markdown("### Ask questions about the podcast")
        
        chatbot = gr.Chatbot(
            label="Conversation",
            height=400,
            bubble_full_width=False
        )
        
        with gr.Row():
            question_input = gr.Textbox(
                label="Your question",
                placeholder="What is the main topic discussed?",
                scale=4
            )
            ask_btn = gr.Button("Ask", variant="primary", scale=1)
        
        clear_btn = gr.Button("Clear Chat")
        
        # Question examples
        gr.Examples(
            examples=[
                "What is the main topic of this podcast?",
                "What are the key takeaways?",
                "Can you summarize the discussion?",
            ],
            inputs=question_input
        )
    
    # Tab 3: Generate Notes
    with gr.Tab("Generate Notes") as notes_tab:
        gr.Markdown("### Generate structured notes from the entire podcast")
        
        generate_notes_btn = gr.Button("Generate Notes", variant="primary")
        notes_output = gr.Markdown(label="Generated Notes")
        
        download_btn = gr.Button("Download Notes")
    
    # Initially hide Q&A and Notes tabs until podcast is processed
    qa_tab_state = gr.State(False)
    notes_tab_state = gr.State(False)
    
    # Event handlers
    process_btn.click(
        fn=process_podcast,
        inputs=[url_input],
        outputs=[status_output, qa_tab_state, notes_tab_state]
    )
    
    ask_btn.click(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",  # Clear input after asking
        outputs=[question_input]
    )
    
    question_input.submit(  # Also trigger on Enter key
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[question_input]
    )
    
    clear_btn.click(
        lambda: [],
        outputs=[chatbot]
    )
    
    generate_notes_btn.click(
        fn=generate_notes,
        outputs=[notes_output]
    )
    
    # Info footer
    gr.Markdown("""
    ---
    ### About
    This app uses:
    - **LangChain** for RAG orchestration
    - **HuggingFace** for LLM inference (Phi-2)
    - **Sentence Transformers** for embeddings
    - **Qdrant** for vector storage
    
    Built by Akanksha | [GitHub](https://github.com/akanksha-th)
    """)


# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # For HuggingFace Spaces
        server_port=7860,        # Default HF Spaces port
        share=False
    )