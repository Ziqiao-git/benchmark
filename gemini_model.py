# chat_models/gemini_model.py

import os
from dotenv import load_dotenv
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from base import ChatModelInterface

# Optionally load .env variables if not done in your main script.
load_dotenv()

class GeminiChatModel(ChatModelInterface):
    def __init__(
        self,
        gemini_api_key: str = None,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.2,
        max_tokens: int = None,
        **model_kwargs
    ):
        """
        Wrapper around Google Generative AI (Gemini) via langchain-google-genai.

        :param gemini_api_key: Your Google API key. If None, reads from env var GEMINI_API_KEY.
        :param model: Model name (e.g. "gemini-2.0-flash").
        :param temperature: Sampling temperature.
        :param max_tokens: Optional max tokens for the output.
        :param model_kwargs: Additional arguments for ChatGoogleGenerativeAI.
        """
        if gemini_api_key is None:
            gemini_api_key = os.getenv("GEMINI_API_KEY", "")

        if not gemini_api_key:
            raise ValueError(
                "No Gemini API key found. Provide gemini_api_key or set GEMINI_API_KEY in your environment."
            )

        self._chat = ChatGoogleGenerativeAI(
            google_api_key=gemini_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **model_kwargs
        )

    def generate_messages(self, messages: list[tuple[str, str]]) -> str:
        """
        Multi-message (system/human) chat usage.
        :param messages: a list of (role, content) tuples, e.g. [("system", "..."), ("human", "...")]
        :return: The model's final text response as a string.
        """
        response = self._chat.invoke(messages)
        return response.content


if __name__ == "__main__":
    print("=== Gemini Model Test ===")

    # Instantiate the wrapper without providing an API key (it will read from env var GEMINI_API_KEY)
    gemini_model = GeminiChatModel(
        model="gemini-2.0-flash",
        temperature=0.2
    )

    msg = [
        ("system", "You are a helpful translator. Respond only in French."),
        ("human", "I love programming.")
    ]
    response = gemini_model.generate_messages(msg)
    print("Response:", response)
    print("=== End of Test ===")
