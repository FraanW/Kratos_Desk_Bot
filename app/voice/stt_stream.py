import numpy as np
from faster_whisper import WhisperModel
from app.config import settings
from app.core.logger import logger
import asyncio

class StreamingTranscriber:
    def __init__(self):
        logger.info("Initializing Whisper model: {} on {}", settings.WHISPER_MODEL, settings.WHISPER_DEVICE)
        try:
            self.model = WhisperModel(
                settings.WHISPER_MODEL,
                device=settings.WHISPER_DEVICE,
                compute_type=settings.WHISPER_COMPUTE_TYPE
            )
        except Exception as e:
            logger.warning("Failed to load Whisper on GPU: {}. Falling back to CPU.", e)
            self.model = WhisperModel(
                settings.WHISPER_MODEL,
                device="cpu",
                compute_type="int8"
            )

    async def transcribe_chunk(self, audio_buffer: np.ndarray):
        from app.core.state_manager import state_manager
        if not state_manager.is_listening():
            return ""

        try:
            return await self._transcribe(audio_buffer)
        except Exception as e:
            if "cublas" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning("CUDA error detected: {}. Falling back to CPU for STT.", e)
                # Re-initialize on CPU
                self.model = WhisperModel(
                    settings.WHISPER_MODEL,
                    device="cpu",
                    compute_type="int8"
                )
                return await self._transcribe(audio_buffer)
            else:
                logger.error("Transcription error: {}", e)
                return ""

    async def _transcribe(self, audio_buffer: np.ndarray):
        from app.core.state_manager import state_manager
        # faster-whisper is blocking, run in executor
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None, 
            lambda: self.model.transcribe(audio_buffer, beam_size=5)
        )
        
        text = ""
        for segment in segments:
            text += segment.text
        
        result = text.strip()
        if not result:
            return ""

        lower_result = result.lower()

        # Wake word detection logic
        if state_manager.is_wake_listening():
            found_wake = any(ww in lower_result for ww in settings.WAKE_WORDS)
            if not found_wake:
                return "" # Ignore non-wake speech
            logger.info(f"Wake word detected in: '{result}'")
        
        # Stop-word detection (only if not in wake mode)
        else:
            from app.core.control_signals import trigger_stop
            for sw in settings.STOP_WORDS:
                if sw in lower_result:
                    logger.info(f"Stop word detected: '{sw}' in '{result}'")
                    trigger_stop()
                    break
                    
        return result

# Singleton
_stt_service = None

def get_stt_service():
    global _stt_service
    if _stt_service is None:
        _stt_service = StreamingTranscriber()
    return _stt_service
