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
from app.core.state_manager import state_manager, AgentState

class KratosOrchestrator:
    def __init__(self):
        self.mic = MicrophoneStream()
        self.stt = get_stt_service()
        self.tts = get_tts_service()
        self.db = SessionLocal()
        self.journal = JournalService(self.db)
        self.silence_timer = 0
        self.is_running = False
        
        # New for upgrade
        self.active_task = None  # Current LLM+TTS pipeline task
        self.last_messages = []  # Message history for repetition guard
        self.max_history = 3
        
        # Start in booting state for announcement
        state_manager.set_state(AgentState.BOOTING)

    async def run(self):
        self.is_running = True
        
        # 0. Startup Announcement
        try:
            state_manager.set_state(AgentState.SPEAKING)
            from app.core.control_signals import clear_stop
            clear_stop()
            await self.tts.speak_sentence("Kratos is online.")
        except Exception as e:
            logger.error(f"Startup announcement failed: {e}")
        finally:
            state_manager.set_state(AgentState.WAKE_LISTENING)

        logger.info("Kratos Orchestrator started. Waiting for wake word...")
        
        audio_buffer = []
        
        # Background task to monitor stop signal
        asyncio.create_task(self._interruption_monitor())
        
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
                    
                # If we were in wake mode and got text (meaning wake word detected)
                if state_manager.is_wake_listening():
                    state_manager.set_state(AgentState.ACTIVE_LISTENING)
                    # We continue so that we don't return to wake mode before processing

                # Set processing state
                state_manager.set_state(AgentState.PROCESSING)
                logger.info("User: {}", text)
                
                # Echo Filter
                if self._is_echo(text):
                    state_manager.set_state(AgentState.LISTENING)
                    continue

                # Repetition Guard
                if self._check_repetition(text):
                    logger.info("Repetitive input detected. Ignoring.")
                    state_manager.set_state(AgentState.WAKE_LISTENING)
                    continue
                
                # Check intent: journaling or conversation
                journal_keywords = ["journal", "record", "remember", "write down", "log"]
                is_journaling = any(kw in text.lower() for kw in journal_keywords)
                
                # Clear any existing stop signal before starting new response
                from app.core.control_signals import clear_stop
                clear_stop()
                
                # Cancel previous active task if it exists
                if self.active_task and not self.active_task.done():
                    self.active_task.cancel()
                
                if is_journaling:
                    self.active_task = asyncio.create_task(self.handle_journal(text))
                else:
                    self.active_task = asyncio.create_task(self.handle_conversation(text))

    async def _interruption_monitor(self):
        """Monitors for the stop signal and cancels active tasks."""
        from app.core.control_signals import wait_for_stop, clear_stop
        while self.is_running:
            await wait_for_stop()
            logger.warning("INTERRUPTION DETECTED. Cancelling tasks...")
            
            if self.active_task and not self.active_task.done():
                self.active_task.cancel()
                self.tts.clear_queue() # Clear any pending speech
                try:
                    await self.active_task
                except asyncio.CancelledError:
                    logger.info("Active task cancelled successfully.")
            
            # Reset
            logger.info("System reset after interruption.")
            clear_stop()
            state_manager.set_state(AgentState.WAKE_LISTENING)

    def _check_repetition(self, text: str) -> bool:
        """Simple repetition guard logic."""
        self.last_messages.append(text.lower().strip())
        if len(self.last_messages) > self.max_history:
            self.last_messages.pop(0)
            
        if len(self.last_messages) == self.max_history:
            # If the last 3 messages are the same and relatively short
            if len(set(self.last_messages)) == 1 and len(text) < 30:
                return True
        return False

    def _is_echo(self, text: str) -> bool:
        """Checks if the transcribed text is an echo of Kratos's last output."""
        last_output = state_manager.get_last_bot_output().lower()
        if not last_output:
            return False
            
        text_lower = text.lower().strip()
        # Remove punctuation for better comparison
        import string
        text_clean = text_lower.translate(str.maketrans('', '', string.punctuation))
        last_clean = last_output.translate(str.maketrans('', '', string.punctuation))
        
        words = text_clean.split()
        if not words: return False
        
        # If the transcript is a subset of the last bot output
        if text_clean in last_clean or last_clean in text_clean:
            logger.info("Direct echo match detected.")
            return True
            
        matches = sum(1 for w in words if w in last_clean)
        overlap = matches / len(words)
        
        if overlap > 0.7:
            logger.info(f"Fuzzy echo detected (overlap={overlap:.2f})")
            return True
            
        return False

    async def handle_journal(self, text: str):
        try:
            # Background task so we don't block response
            asyncio.create_task(self.journal.add_entry(text))
            
            state_manager.set_state(AgentState.SPEAKING)
            await self.tts.speak_sentence("I have recorded your words. They are etched in memory.")
        except asyncio.CancelledError:
            logger.info("Journal response cancelled.")
        finally:
            state_manager.set_state(AgentState.WAKE_LISTENING)

    async def handle_conversation(self, text: str):
        try:
            # 1. Retrieve memories
            memories = await self.journal.search_memory(text)
            weekly = await self.journal.get_latest_weekly_summary()
            
            # 2. Build prompt
            prompt = build_prompt(text, memories, weekly)
            
            # 3. Stream LLM -> TTS
            state_manager.set_state(AgentState.SPEAKING)
            token_stream = stream_llm_response(prompt)
            await self.tts.stream_sentences(token_stream)
        except asyncio.CancelledError:
            logger.info("Conversation task cancelled.")
        except Exception as e:
            logger.error("Error in handle_conversation: {}", e)
        finally:
            # Always return to wake listening
            state_manager.set_state(AgentState.WAKE_LISTENING)

    async def stop(self):
        self.is_running = False
        if hasattr(self.tts, 'worker_task'):
            self.tts.worker_task.cancel()
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()
        self.mic.stop()
        self.db.close()
