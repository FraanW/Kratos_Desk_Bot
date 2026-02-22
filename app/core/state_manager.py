from enum import Enum, auto
from app.core.logger import logger

class AgentState(Enum):
    BOOTING = auto()
    WAKE_LISTENING = auto()
    ACTIVE_LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()

class StateManager:
    def __init__(self):
        self._state = AgentState.BOOTING
        self._last_bot_output = ""

    @property
    def state(self) -> AgentState:
        return self._state

    def set_state(self, new_state: AgentState):
        if self._state != new_state:
            logger.info(f"Agent state changed: {self._state.name} -> {new_state.name}")
            self._state = new_state

    def is_listening(self) -> bool:
        """Returns true if the agent is listening for anything (wake or active)."""
        return self._state in [AgentState.WAKE_LISTENING, AgentState.ACTIVE_LISTENING]

    def is_wake_listening(self) -> bool:
        return self._state == AgentState.WAKE_LISTENING

    def is_active_listening(self) -> bool:
        return self._state == AgentState.ACTIVE_LISTENING

    def is_processing(self) -> bool:
        return self._state == AgentState.PROCESSING

    def is_speaking(self) -> bool:
        return self._state == AgentState.SPEAKING

    def set_last_bot_output(self, text: str):
        self._last_bot_output = text

    def get_last_bot_output(self) -> str:
        return self._last_bot_output

# Global Singleton
state_manager = StateManager()
