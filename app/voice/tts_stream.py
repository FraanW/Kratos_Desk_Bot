import pyttsx3
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.core.logger import logger
import re

class StreamingTTS:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._sentence_regex = re.compile(r'[^.!?]+[.!?]')
        
    def _speak(self, text: str):
        try:
            engine = pyttsx3.init()
            # Kratos style: slow, deep
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            voices = engine.getProperty('voices')
            # Select a male voice if available
            for voice in voices:
                if "male" in voice.name.lower() or "david" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.error("TTS Error: {}", e)

    async def speak_sentence(self, text: str):
        if not text.strip():
            return
        logger.info("Speaking: {}", text)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._speak, text)

    async def stream_sentences(self, token_generator):
        """
        Buffers tokens and speaks as soon as a sentence is complete.
        """
        buffer = ""
        async for token in token_generator:
            buffer += token
            
            # Find complete sentences
            match = self._sentence_regex.search(buffer)
            if match:
                sentence = match.group(0)
                buffer = buffer[match.end():]
                # Speak in background without blocking token stream
                asyncio.create_task(self.speak_sentence(sentence))
        
        # Speak remaining buffer
        if buffer.strip():
            await self.speak_sentence(buffer)

# Singleton
_tts_service = None

def get_tts_service():
    global _tts_service
    if _tts_service is None:
        _tts_service = StreamingTTS()
    return _tts_service
