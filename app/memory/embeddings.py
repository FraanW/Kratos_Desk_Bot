from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import settings
from app.core.logger import logger

class EmbeddingService:
    def __init__(self):
        logger.info("Loading embedding model: {}", settings.EMBEDDING_MODEL)
        # Using CPU for embeddings to save VRAM for LLM/Whisper
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")
        
    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

# Singleton instance
_embedding_service = None

def get_embedding_service():
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
