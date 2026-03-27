from deepeval.models.base_model import DeepEvalBaseLLM
from api.core.config import api_settings
from groq import Groq
from aiolimiter import AsyncLimiter
import asyncio

# qwen3-32b free tier: 6000 TPM — use ~4 req/min to stay safe
# each eval call is ~500-1000 tokens, so 4 req/min ≈ 4000 tokens/min headroom
GROQ_RATE_LIMITER = AsyncLimiter(max_rate=4, time_period=60)

class GroqEvalLLM(DeepEvalBaseLLM):
    def __init__(self):
        settings = api_settings()
        self.client = Groq(api_key=settings.groq.api_key)
        self.model = settings.groq.evaluation_model

    def load_model(self):
        """Required by DeepEvalBaseLLM"""
        return self.client

    def generate(self, prompt: str) -> str:
        strict_prompt = f"/nothink\nYou MUST return output in valid JSON. Do not include any text before or after the JSON object.\n\n{prompt}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": strict_prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    
    async def a_generate(self, prompt: str) -> str:
        async with GROQ_RATE_LIMITER:
            return await asyncio.to_thread(self.generate, prompt)
    
    def get_model_name(self) -> str:
        return f"groq:{self.model}"
    
# if __name__ == "__main__":
#     serv = GroqEvalLLM()
#     print(serv.client.models.list())