import os
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Project Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Audio Settings
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_SIZE: int = 1024
    SILENCE_THRESHOLD: float = 0.01
    SILENCE_DURATION: float = 1.5  # Seconds
    
    # STT Settings (Whisper)
    WHISPER_MODEL: str = "small"
    WHISPER_DEVICE: str = "cuda"  # Will fallback to cpu if cuda not available
    WHISPER_COMPUTE_TYPE: str = "float16"
    
    # LLM Settings (Ollama)
    OLLAMA_URL: str = "http://localhost:11434/api/generate"
    OLLAMA_MODEL: str = "llama3.1"
    MAX_TOKENS: int = 200
    TEMPERATURE: float = 0.7
    NUM_CTX: int = 2048
    
    # TTS Settings (Coqui XTTS)
    XTTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    SPEAKER_WAV: str = "D:/Kratos_Desk/kratos_15s_sample.wav"
    
    # Memory & Database
    DATABASE_URL: str = f"sqlite:///{DATA_DIR}/kratos.db"
    FAISS_INDEX_PATH: str = str(DATA_DIR / "faiss.index")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Agent Logic
    JOURNAL_COMPRESSION_THRESHOLD: int = 25
    STOP_WORDS: list[str] = ["stop", "silence", "enough", "shut up"]
    WAKE_WORDS: list[str] = ["kratos", "hey kratos"]
    
    class Config:
        env_file = ".env"

settings = Settings()

# Ensure speaker wav exists (warn if not)
if not Path(settings.SPEAKER_WAV).exists():
    print(f"WARNING: Speaker WAV not found at {settings.SPEAKER_WAV}")

# Ensure data directory exists
os.makedirs(settings.DATA_DIR, exist_ok=True)
