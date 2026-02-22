import aiohttp
import json
from app.config import settings
from app.core.logger import logger

class Summarizer:
    async def summarize_entry(self, text: str) -> str:
        prompt = f"Summarize this journal entry in 10-15 words, focusing on the core emotion and event. Be concise and Kratos-like:\n\n{text}"
        return await self._call_ollama(prompt)

    async def summarize_weekly(self, texts: list[str]) -> str:
        combined = "\n---\n".join(texts)
        prompt = f"Provide a powerful weekly summary for these journal entries. Highlight progress and areas of struggle. Keep it short (2-3 sentences):\n\n{combined}"
        return await self._call_ollama(prompt)

    async def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": 1024
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(settings.OLLAMA_URL, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("response", "").strip()
                    else:
                        logger.error("Ollama summary request failed: {}", resp.status)
                        return ""
        except Exception as e:
            logger.error("Error calling Ollama for summary: {}", e)
            return ""

# Singleton
_summarizer = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = Summarizer()
    return _summarizer
