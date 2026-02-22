import os
import torch
import numpy as np
import sounddevice as sd
from TTS.api import TTS
from app.config import settings
from app.core.logger import logger
from app.core.control_signals import is_stopped
from app.core.state_manager import state_manager
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

class CoquiTTS:
    def __init__(self):
        logger.info("Loading Coqui XTTS v2 model: {}", settings.XTTS_MODEL)
        # Load model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # We'll use half precision if available to save VRAM
        self.tts = TTS(settings.XTTS_MODEL).to(device)
        
        # Verify speaker wav
        if not os.path.exists(settings.SPEAKER_WAV):
            logger.error("Speaker WAV not found at: {}. Using default voice.", settings.SPEAKER_WAV)
            self.speaker_wav = None
        else:
            self.speaker_wav = settings.SPEAKER_WAV
            
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.sample_rate = 24000  # XTTS v2 default sample rate
        
        # New for upgrade: Sequential playback queue
        self.playback_queue = asyncio.Queue()
        self.worker_task = asyncio.create_task(self._playback_worker())

    async def _playback_worker(self):
        """Worker that pulls sentences from the queue and speaks them sequentially."""
        logger.info("TTS playback worker started.")
        while True:
            try:
                text = await self.playback_queue.get()
                if text is None: # Sentinel for shutdown
                    break
                
                await self.speak_sentence(text)
                self.playback_queue.task_done()
            except Exception as e:
                logger.error("Error in TTS playback worker: {}", e)
                # Keep worker alive
                try: self.playback_queue.task_done() 
                except: pass

    def clear_queue(self):
        """Clears the playback queue and stops current playback."""
        logger.info("Clearing TTS playback queue.")
        # Re-initialize the queue to drop old items
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
                self.playback_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        # Stop immediate playback if any
        try:
            sd.stop()
        except:
            pass

    def _clean_sentence(self, text: str) -> str:
        """Strips quotes, extra whitespace, and normalizes the sentence."""
        if not text:
            return ""
        
        cleaned = text.strip()
        # Remove wrapping quotes
        cleaned = cleaned.strip('"\'')
        # Normalize internal whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def _is_valid_sentence(self, text: str) -> bool:
        """Validates if a sentence is worth speaking (length, content)."""
        if not text:
            return False
            
        # Reject if too short (e.g. isolated quotes or single letters)
        # But allow "No.", "Yes." which are >= 3 chars
        if len(text) < 3:
            return False
            
        # Must contain at least one alphanumeric character
        if not any(c.isalnum() for c in text):
            return False
            
        return True

    def _play_audio(self, wav_data: list[float]):
        """Callback to play audio in a background thread."""
        try:
            if is_stopped():
                return
            
            audio_array = np.array(wav_data, dtype=np.float32)
            sd.play(audio_array, samplerate=self.sample_rate)
            sd.wait() # Wait for playback to finish
        except Exception as e:
            logger.error("Audio playback error: {}", e)

    async def speak_sentence(self, text: str):
        """Generates and plays speech for a single sentence."""
        if not text.strip() or is_stopped():
            return

        logger.info("Generating XTTS: '{}'", text)
        
        # Generation is blocking, run in executor
        loop = asyncio.get_event_loop()
        try:
            # Generate wav in-memory
            wav = await loop.run_in_executor(
                self.executor,
                lambda: self.tts.tts(
                    text=text,
                    speaker_wav=self.speaker_wav,
                    language="en"
                )
            )
            
            # Play in background thread
            if not is_stopped():
                await loop.run_in_executor(None, self._play_audio, wav)
                
        except Exception as e:
            logger.error("XTTS Generation error: {}", e)

    async def stream_sentences(self, token_generator):
        """Buffers tokens and speaks as soon as a sentence is complete."""
        sentence_regex = re.compile(r'[^.!?]+[.!?]')
        buffer = ""
        full_response = ""
        
        async for token in token_generator:
            if is_stopped():
                logger.info("TTS stream interrupted.")
                self.clear_queue()
                break
                
            buffer += token
            
            # Find complete sentences
            match = sentence_regex.search(buffer)
            if match:
                sentence = match.group(0)
                buffer = buffer[match.end():]
                
                # SANITATION & VALIDATION
                cleaned = self._clean_sentence(sentence)
                if self._is_valid_sentence(cleaned):
                    # Put in queue for sequential playback
                    await self.playback_queue.put(cleaned)
                    full_response += cleaned + " "
                else:
                    logger.debug("Skipping invalid/empty fragment: '{}'", sentence)
        
        # Speak remaining buffer
        if buffer.strip() and not is_stopped():
            cleaned_rem = self._clean_sentence(buffer)
            if self._is_valid_sentence(cleaned_rem):
                await self.playback_queue.put(cleaned_rem)
                full_response += cleaned_rem
            else:
                logger.debug("Skipping invalid/empty remaining fragment: '{}'", buffer)

        # Store full response for echo filtering
        state_manager.set_last_bot_output(full_response.strip())

        # Wait for all queued speech to finish
        await self.playback_queue.join()

# Singleton
_coqui_service = None

def get_tts_service():
    global _coqui_service
    if _coqui_service is None:
        _coqui_service = CoquiTTS()
    return _coqui_service
