import asyncio
import numpy as np
import sounddevice as sd
from app.config import settings
from app.core.logger import logger

class MicrophoneStream:
    def __init__(self):
        self.sample_rate = settings.SAMPLE_RATE
        self.chunk_size = settings.CHUNK_SIZE
        self.queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop()
        self.active = False

    def _callback(self, indata, frames, time, status):
        if status:
            logger.warning("Sounddevice status: {}", status)
        # Put raw audio into the queue
        self.loop.call_soon_threadsafe(self.queue.put_nowait, indata.copy())

    async def stream(self):
        self.active = True
        logger.info("Starting microphone stream ({} Hz)...", self.sample_rate)
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=settings.CHANNELS,
            blocksize=self.chunk_size,
            callback=self._callback,
            dtype='float32'
        ):
            while self.active:
                chunk = await self.queue.get()
                
                # Check for silence detection (RMS)
                rms = np.sqrt(np.mean(chunk**2))
                is_silent = rms < settings.SILENCE_THRESHOLD
                
                yield chunk, is_silent

    def stop(self):
        self.active = False
        logger.info("Microphone stream stopped.")
