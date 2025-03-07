# chat_models/claude_model.py

from langchain_anthropic import ChatAnthropic
from base import ChatModelInterface
from dotenv import load_dotenv
import os

# # Load environment variables from .env (if you haven't already loaded it elsewhere)
load_dotenv()

class ClaudeChatModel(ChatModelInterface):
    def __init__(
        self,
        anthropic_api_key: str = None,
        model: str = "claude-2",
        temperature: float = 0.0,
        **model_kwargs
    ):
        """
        Wrapper around Anthropic's Claude model via langchain_anthropic.

        :param anthropic_api_key: Your Anthropic API key (sk-ant-...).
                                  If None, will read from env var CLAUDE_API_KEY.
        :param model: Claude model name, e.g. "claude-2" or "claude-instant-1".
        :param temperature: Sampling temperature (0 = deterministic).
        :param model_kwargs: Additional arguments for ChatAnthropic.
        """
        # If the user didn't provide a key, read from environment
        if anthropic_api_key is None:
            anthropic_api_key = os.getenv("CLAUDE_API_KEY")

        if not anthropic_api_key:
            raise ValueError(
                "No CLAUDE_API_KEY found. Provide anthropic_api_key directly "
                "or set CLAUDE_API_KEY in your environment or .env file."
            )

        self._chat = ChatAnthropic(
            anthropic_api_key=anthropic_api_key,
            model=model,
            temperature=temperature,
            **model_kwargs
        )

    def generate_messages(self, messages: list[tuple[str, str]]) -> str:
        """
        Multi-message (system/human) interactions:
        :param messages: a list of (role, content) tuples, e.g. [("system", "..."), ("human", "...")]
        :return: model's final text response
        """
        response = self._chat.invoke(messages)
        return response.content

# Quick test
if __name__ == "__main__":
    print("=== Claude Model Test ===")

    # We'll assume CLAUDE_API_KEY is set in your environment / .env file.
    claude_model = ClaudeChatModel(
        model="claude-3-7-sonnet-20250219",
        temperature=0
    )

    msg = [
        ("system", "You are a helpful translator. Respond only in French."),
        ("human", "I love programming.")
    ]
    response = claude_model.generate_messages(msg)
    print("Response:", response)
    print("=== End of Test ===")
