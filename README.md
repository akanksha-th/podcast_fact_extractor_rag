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

The system is implemented as a LangGraph state machine, where each step is a clearly defined node:

1. **Transcription Node** - Fetches podcast transcripts (YouTube subtitles preferred, audio fallback supported).

2. **Chunk & Embed Node** - Splits transcripts into fixed-size chunks and generates embeddings using Sentence Transformers.

3. **Store Node** - Stores embeddings and text payloads in a local Qdrant vector database.

4. **Query Node (User Input)** - Accepts user questions in a loop until exit / quit is entered.

5. **Retrieve Node** - Performs semantic search in Qdrant to retrieve the most relevant transcript chunks.

6. **Generate Node** - Uses a Retrieval-Augmented prompt and an open-source LLM to generate factual answers only from retrieved context.

7. **Conditional Loop** - The agent continues answering questions until the user exits.
  
</td>
<td>

<img src="graph_images/agent_graph.png" alt="Agent Graph" style="width:100%; max-width:400px; border-radius:10px;"/>

</td>
</tr>
</table>

---

## Tech Stack

- **Python**: 3.11+
- **yt-dlp / Faster-Whisper** - For transcript ingestion
- **Langchain / HuggingFace / Sentence Transformers** - For embeddings
- **LLM (HuggingFace)** - For answer generation
- **Qdrant** - For vector storage and search
- **GPT4All / GGUF-based LLM** - Open-source local inference
_.No proprietary APIs are required. The system runs fully offline after setup.._

---

## LLM Model Setup

This project uses a local, open-source LLM (phi-2, GGUF format) for answer generation. 
The model must be downloaded before running the agent.

Download the model file from HuhggingFace using the link: [Phi-2 GGUF Model](https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_0.gguf?download=true)

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
python -m src.agents.qna_agent
```

---

## License
MIT License