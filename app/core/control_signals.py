import asyncio
from app.core.logger import logger

# Global event for interruption
_stop_signal = asyncio.Event()

def trigger_stop():
    """Signals that the current output should be interrupted."""
    if not _stop_signal.is_set():
        logger.info("Interruption triggered!")
        _stop_signal.set()

def clear_stop():
    """Clears the interruption signal."""
    if _stop_signal.is_set():
        _stop_signal.clear()

def is_stopped() -> bool:
    """Checks if the interruption signal is currently set."""
    return _stop_signal.is_set()

async def wait_for_stop():
    """Wait until the stop signal is triggered."""
    await _stop_signal.wait()

async def check_interruption():
    """Raises a CancelledError if the stop signal is set."""
    if is_stopped():
        raise asyncio.CancelledError("Interrupted by stop signal")
