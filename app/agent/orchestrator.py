import asyncio
import numpy as np
from app.voice.audio_stream import MicrophoneStream
from app.voice.stt_stream import get_stt_service
from app.voice.tts_stream import get_tts_service
from app.llm.ollama_stream import stream_llm_response
from app.llm.prompt_builder import build_prompt
from app.journal.journal_service import JournalService
from app.memory.database import SessionLocal
from app.core.logger import logger
from app.config import settings

class KratosOrchestrator:
    def __init__(self):
        self.mic = MicrophoneStream()
        self.stt = get_stt_service()
        self.tts = get_tts_service()
        self.db = SessionLocal()
        self.journal = JournalService(self.db)
        self.silence_timer = 0
        self.is_running = False

    async def run(self):
        self.is_running = True
        logger.info("Kratos Orchestrator started. Speak now.")
        
        audio_buffer = []
        
        async for chunk, is_silent in self.mic.stream():
            if not self.is_running:
                break
                
            audio_buffer.append(chunk)
            
            if is_silent:
                self.silence_timer += len(chunk) / settings.SAMPLE_RATE
            else:
                self.silence_timer = 0
            
            # End of speech detection
            if self.silence_timer >= settings.SILENCE_DURATION and len(audio_buffer) > 5:
                # Combine and flatten buffer (remove channels dimension)
                full_audio = np.concatenate(audio_buffer).flatten()
                audio_buffer = [] # Reset buffer
                self.silence_timer = 0
                
                # Transcribe
                text = await self.stt.transcribe_chunk(full_audio)
                if not text:
                    continue
                    
                logger.info("User: {}", text)
                
                # Check intent: journaling or conversation
                # Simple keyword heuristic as requested
                journal_keywords = ["journal", "record", "remember", "write down", "log"]
                is_journaling = any(kw in text.lower() for kw in journal_keywords)
                
                if is_journaling:
                    await self.handle_journal(text)
                else:
                    await self.handle_conversation(text)

    async def handle_journal(self, text: str):
        # Background task so we don't block response
        asyncio.create_task(self.journal.add_entry(text))
        await self.tts.speak_sentence("I have recorded your words. They are etched in memory.")

    async def handle_conversation(self, text: str):
        # 1. Retrieve memories
        memories = await self.journal.search_memory(text)
        weekly = await self.journal.get_latest_weekly_summary()
        
        # 2. Build prompt
        prompt = build_prompt(text, memories, weekly)
        
        # 3. Stream LLM -> TTS
        token_stream = stream_llm_response(prompt)
        await self.tts.stream_sentences(token_stream)

    def stop(self):
        self.is_running = False
        self.mic.stop()
        self.db.close()
