import os
from dotenv import load_dotenv
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from role_models import CompleteChatModel

load_dotenv()

class GeminiChatModel(CompleteChatModel):
    """
    Google's Gemini implementation using LangChain.
    Inherits all JSON generation methods from CompleteChatModel.
    """

    def __init__(
        self,
        gemini_api_key: str = None,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.2,
        max_tokens: int = None,
        **model_kwargs
    ):
        """
        Initialize the Gemini model.
        
        Args:
            gemini_api_key: Google API key, defaults to GEMINI_API_KEY env var
            model: Model name, e.g. "gemini-2.0-flash"
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            model_kwargs: Additional kwargs passed to ChatGoogleGenerativeAI
        
        Raises:
            ValueError: If no API key is provided or found in env vars
        """
        if gemini_api_key is None:
            gemini_api_key = os.getenv("GEMINI_API_KEY", "")

        if not gemini_api_key:
            raise ValueError(
                "No Gemini API key found. Provide gemini_api_key directly "
                "or set GEMINI_API_KEY in your environment or .env file."
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
        Send messages to Gemini and get a text response.
        
        Args:
            messages: List of (role, content) tuples
            
        Returns:
            Text response from the model
        """
        response = self._chat.invoke(messages)
        return response.content


if __name__ == "__main__":
    print("=== Gemini Model Test ===")
    gemini_model = GeminiChatModel(
        model="gemini-2.0-flash",
        temperature=0.2
    )
    msgs = [
        ("system", "You are a helpful translator. Respond only in French."),
        ("human", "I love programming.")
    ]
    response = gemini_model.generate_messages(msgs)
    print("Response:", response)
    print("=== End of Test ===")