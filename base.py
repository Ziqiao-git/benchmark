# chat_models/base.py
from abc import ABC, abstractmethod

class ChatModelInterface(ABC):
    """A standard interface for any chat model."""

    @abstractmethod
    def generate_messages(self, messages: list[tuple[str, str]]) -> str:
        """Send a prompt to the model and return its text response."""
        pass
