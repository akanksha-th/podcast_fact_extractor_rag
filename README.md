# Podcast Fact Extractor RAG
A lightweight Retrieval-Augmented-Generation (RAG) system design to extract factual insights from podcast transcripts.

---
<table>
<tr>
<td>
    
## Project Overview

This project implements a **mini RAG pipeline** to:

1. Transcribe the podcast.
2. Chunk and embed the transcript data.
3. Retrieve relevant information for a user query.
4. Generate concise, accurate answers using an LLM.

## How It Works

1. **Ingest Podcast** - Takes the **YouTube podcast url** as input and fetches its transcripts.
2. **Chunking and Embedding** - Breaks transcripts into chunks and generates embeddings.
3. **Store and Search** - Store embeddings in a vector database (Qdrant) for semantic search.
4. **Answer Queries** - User asks a question —> relevant chunks retrieved —> LLM generates an answer.
  
</td>
<td>

<img src="graph_images/agent_graph.png" alt="Agent Graph" style="width:100%; max-width:400px; border-radius:10px;"/>

</td>
</tr>
</table>

---

## Tech Stack

- **Python**
- **Faster-Whisper**
- **Langchain / HuggingFace / Sentence Transformers** - For embeddings
- **LLM (HuggingFace)** - For answer generation
- **Qdrant** - For vector storage and search
- _.Deployment (still pending)._

---

## Usage
```
# Clone the repo
git clone <repo_link>
cd podcast_fact_extractor_rag

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python -m src.agents.qna_agent
```
