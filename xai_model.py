# chat_models/xai_model.py

import os
from dotenv import load_dotenv
from langchain_xai import ChatXAI
from base import ChatModelInterface

# Optionally load dotenv if you haven't done so elsewhere
# This will read environment variables from a .env file
load_dotenv()

class XAIChatModel(ChatModelInterface):
    def __init__(
        self, 
        xai_api_key: str = None, 
        model: str = "grok-2", 
        temperature: float = 0.2,
        **model_kwargs
    ):
        """
        Wrapper around xAI's ChatXAI model.

        :param xai_api_key: Your xAI API key. If None, will look for XAI_API_KEY in the environment.
        :param model: xAI model name (e.g. 'grok-2', 'grok-2-1212', etc.).
        :param temperature: Sampling temperature, if supported.
        :param model_kwargs: Additional arguments for ChatXAI.
        """

        # If no key was provided, read from the environment
        if xai_api_key is None:
            xai_api_key = os.getenv("XAI_API_KEY", "")

        if not xai_api_key:
            raise ValueError(
                "No xAI API key found. Please provide xai_api_key or set XAI_API_KEY in your .env or environment."
            )

        self._chat = ChatXAI(
            xai_api_key=xai_api_key,
            model=model,
            temperature=temperature,
            **model_kwargs
        )
    
    def generate_messages(self, messages: list[tuple[str, str]]) -> str:
        """
        Multi-message (system/human) chat-style usage:
        
        :param messages: a list of (role, content) tuples, e.g. 
                         [("system", "..."), ("human", "...")]
        :return: The final string response from the xAI model
        """
        response = self._chat.invoke(messages)
        return response.content

if __name__ == "__main__":
    # Simple test usage

    print("=== XAI Model Test ===")

    # Instantiate the wrapper without providing an API key
    # (It will read from XAI_API_KEY in environment or .env)
    xai_model = XAIChatModel(
        model="grok-2",   # or "grok-2-1212", etc.
        temperature=0.1
    )

    # Prepare multi-role messages
    msg = [
        ("system", "You are a helpful translator. Respond only in French."),
        ("human", "I love programming.")
    ]
    response = xai_model.generate_messages(msg)
    print("Response:", response)
    print("=== End of Test ===")
