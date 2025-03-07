# chat_models/deepseek_model.py

import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from base import ChatModelInterface

# Optionally load environment variables from .env if not done in main script
load_dotenv()

class DeepSeekChatModel(ChatModelInterface):
    def __init__(
        self,
        deepseek_api_key: str = None,
        model: str = "deepseek-reasoner",
        temperature: float = 0.2,
        **model_kwargs
    ):
        """
        Wrapper around DeepSeek using langchain's ChatOpenAI with custom openai_api_base.

        :param deepseek_api_key: Your DeepSeek API key (sk-...). If None, reads from env var DEEPSEEK_API_KEY.
        :param model: The DeepSeek model name, e.g. "deepseek-reasoner".
        :param temperature: Sampling temperature.
        :param model_kwargs: Additional arguments for ChatOpenAI.
        """
        if deepseek_api_key is None:
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")

        if not deepseek_api_key:
            raise ValueError(
                "No DeepSeek API key found. Provide deepseek_api_key or set DEEPSEEK_API_KEY in your environment."
            )

        self._chat = ChatOpenAI(
            openai_api_key=deepseek_api_key,
            openai_api_base="https://api.deepseek.com",  # The DeepSeek endpoint
            model_name=model,
            temperature=temperature,
            **model_kwargs
        )

    def generate_messages(self, messages: list[tuple[str, str]]) -> str:
        """
        Accepts a list of (role, content) tuples, e.g.:
            [("system", "..."), ("human", "Hello"), ...]

        Returns the model's final text response as a string.
        """
        response = self._chat.invoke(messages)
        return response.content


if __name__ == "__main__":
    # Example usage / test
    print("=== DeepSeek Model Test ===")

    # Instantiate the DeepSeek wrapper without providing a key
    # (It will look for DEEPSEEK_API_KEY in your environment.)
    ds_model = DeepSeekChatModel(
        model="deepseek-reasoner",
        temperature=0.2
    )

    # Build some test messages
    msg = [
        ("system", "You are a helpful assistant that speaks only in French."),
        ("human", "I love programming!")
    ]

    # Call the multi-message method
    response = ds_model.generate_messages(msg)

    print("Messages:", msg)
    print("Response:", response)
    print("=== End of Test ===")
