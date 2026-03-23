from langchain_groq import ChatGroq
from api.core.config import api_settings
from api.utils.prompts import build_rag_prompt, build_map_prompt, build_reduce_prompt
from api.core.metrics import groq_latency
import time


settings = api_settings()

class QueryLLMService:
    def __init__(self, context: list[dict], history: dict, question: str):
        self.context = context
        self.history = history
        self.question = question
        self.rag_prompt = build_rag_prompt(self.context, self.history, self.question)

        self.query_groq = ChatGroq(
            groq_api_key=settings.groq.api_key,
            model=settings.groq.ask_model,
            temperature=0.0,
            max_retries=2
        )

    async def get_answer(self):
        try:
            start = time.time()
            response = await self.query_groq.ainvoke(self.rag_prompt)
            groq_latency.observe(time.time() - start)
            return response.content
        except Exception as e:
            raise RuntimeError(f"LLM Service is temporarily unavailable: {e}")
        

class NotesLLMService:
    def __init__(self):

        self.notes_groq = ChatGroq(
            groq_api_key=settings.groq.api_key,
            model=settings.groq.notes_model,
            temperature=0.0,
            max_retries=2
        )
        
    async def map_summary(self, context: str):
        try:
            prompt = build_map_prompt(context)
            start = time.time()
            response = await self.notes_groq.ainvoke(prompt)
            groq_latency.observe(time.time() - start)
            return response.content
        except Exception as e:
            raise RuntimeError("LLM Service is temporarily unavailable.")
        
    async def reduce_summary(self, context: str):
        try:
            prompt = build_reduce_prompt(context)
            start = time.time()
            response = await self.notes_groq.ainvoke(prompt)
            groq_latency.observe(time.time() - start)
            return response.content
        except Exception as e:
            raise RuntimeError("LLM Service is temporarily unavailable.")