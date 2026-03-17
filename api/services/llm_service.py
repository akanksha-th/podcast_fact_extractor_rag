from langchain_groq import ChatGroq
from api.core.config import api_settings
from api.utils.prompts import build_rag_prompt

settings = api_settings()

class LLMService:
    def __init__(self, context: list[dict], history: dict, question: str):
        self.context = context
        self.history = history
        self.question = question
        self.rag_prompt = build_rag_prompt(self.context, self.history, self.question)

        self.groq = ChatGroq(
            model=settings.groq.ask_model,
            temperature=0.0,
            max_retries=2
        )
        
    async def get_answer(self):
        try:
            response = await self.groq.ainvoke(self.rag_prompt)
            return response.content
        except Exception as e:
            raise RuntimeError("LLM Service is temporarily unavailable.")