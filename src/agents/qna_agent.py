from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from src.core.ingestion import get_transcription


class ExtractorState(TypedDict):
    url: str
    transcripts: str


def transcription(state: ExtractorState) -> ExtractorState:
    """Extracts video transcripts"""
    url = state["url"]
    transcripts = get_transcription(url)

    state["transcripts"] = transcripts
    print(f"[Transcription] Transcription has been done successfully.")
    return state



def build_graph() -> StateGraph:
    graph = StateGraph(ExtractorState)

    graph.add_node("transcription", transcription)

    graph.add_edge(START, "transcription")

    return graph.compile()


if __name__ == "__main__":
    url = ""
    app = build_graph()
    result = app.invoke({
        "url": url,
        "transcripts": ""
    })