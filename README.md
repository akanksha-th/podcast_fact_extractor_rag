# Podcast Fact Extractor RAG
A lightweight Retrieval-Augmented-Generation (RAG) system designed to extract factual insights and structures notes from podcast transcripts.

---
<table>
<tr>
<td>
    
## Project Overview

This project implements a **mini RAG pipeline** to:

Transcribing podcasts from YouTube or audio.
2. Chunking and embedding transcript data.
3. Storing embeddings in a vector database.
4. Answering user questions using retrieval-augmented generation.
5. Generating AI-powered, structured notes from podcast transcripts.

## How It Works

The system is implemented as a LangGraph state machine, where each step is a clearly defined node:

1. **Transcription Node** - Fetches podcast transcripts (YouTube subtitles preferred, audio fallback supported).

2. **Chunk & Embed Node** - Splits transcripts into fixed-size chunks and generates embeddings using Sentence Transformers.

3. **Store Node** - Stores embeddings and text payloads in a local Qdrant vector database.

4. **Task Selection Node**  
   Allows the user to choose between:
   - Question Answering (Q&A)
   - AI-powered Notes Generation

5. **Query Node (User Input)** - Accepts user questions in a loop until exit / quit is entered.

6. **Retrieve Node** - Performs semantic search in Qdrant to retrieve the most relevant transcript chunks.

7. **Generate Node** - Uses a Retrieval-Augmented prompt and an open-source LLM to generate answers strictly grounded in retrieved transcript context.

8. **Notes Generation Node (Notes Mode)**  
Generates structured, simplified notes from transcript chunks using a dedicated notes prompt and chunk-wise summarization.
   
9. **Conditional Loop** - The agent continues answering questions until the user exits.
  
</td>
<td>

<img src="graph_images/agent_graph_v1.png" alt="Agent Graph" style="width:100%; max-width:400px; border-radius:10px;"/>

</td>
</tr>
</table>

---

## Tech Stack

- **Python**: 3.11+
- **yt-dlp / Faster-Whisper** - For transcript ingestion
- **Langchain / Hugging Face / Sentence Transformers** - For embeddings
- **Local LLM (GPT4All / GGUF)** - For offline answer and notes generation
- **Qdrant** - For vector storage and search
_No proprietary APIs are required. The system runs fully offline after setup._

---

## LLM Model Setup

This project uses a local, open-source LLM (phi-2, GGUF format) for answer generation. 
The model must be downloaded before running the agent.

Download the model file from Huhgging Face using the link: [Phi-2 GGUF Model](https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_0.gguf?download=true)

After downloading the file, move it to the models folder.

---

## Usage
```
# Clone the repo
git clone <repo_link>
cd podcast_fact_extractor_rag

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python -m src.agents.qna_agent_v0   # Older version - v0
python -m src.agents.qna_and_notes_agent_v1     # v1
```

---

## Versions

v0    —> Simple QnA on CLI
v1.1  —> Multilingual control
v1.2  —> Audio query support
v1.3  —> Multi-source ingestion (Spotify)
v2    —> Knowledge Graph Augmentation

---

## License
MIT License