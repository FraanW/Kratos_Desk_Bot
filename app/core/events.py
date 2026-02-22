from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.logger import logger
from app.config import settings

# Placeholder for model registry or dependency injection
MODELS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing Kratos Stream Engine...")
    
    # In a real implementation, we would preload models here
    # and store them in app.state or a global registry
    
    logger.info("Preloading Whisper model: {}", settings.WHISPER_MODEL)
    # from faster_whisper import WhisperModel
    # MODELS['whisper'] = WhisperModel(...)
    
    logger.info("Preloading Embedding model: {}", settings.EMBEDDING_MODEL)
    # from sentence_transformers import SentenceTransformer
    # MODELS['embeddings'] = SentenceTransformer(...)
    
    logger.info("Kratos Desk is ready.")
    yield
    
    # Shutdown
    logger.info("Shutting down Kratos Stream Engine...")
    # Clean up resources if needed
