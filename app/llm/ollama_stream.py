import aiohttp
import json
from app.config import settings
from app.core.logger import logger

async def stream_llm_response(prompt: str):
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": settings.TEMPERATURE,
            "num_ctx": settings.NUM_CTX,
            "num_predict": settings.MAX_TOKENS
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(settings.OLLAMA_URL, json=payload) as resp:
                if resp.status != 200:
                    logger.error("Ollama streaming error: {}", resp.status)
                    yield "I am silent, for the connection is broken."
                    return

                async for line in resp.content:
                    if not line:
                        continue
                    
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("response", "")
                    if token:
                        yield token
                    
                    if data.get("done"):
                        break
    except Exception as e:
        logger.error("Error streaming from Ollama: {}", e)
        yield "The void consumes my words."
