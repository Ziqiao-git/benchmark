import os
from dotenv import load_dotenv
from langchain_xai import ChatXAI
from role_models import CompleteChatModel

load_dotenv()

class XAIChatModel(CompleteChatModel):
    """
    xAI's implementation using LangChain.
    Inherits all JSON generation methods from CompleteChatModel.
    """

    def __init__(
        self,
        xai_api_key: str = None,
        model: str = "grok-2",
        temperature: float = 0.2,
        **model_kwargs
    ):
        """
        Initialize the xAI model.
        
        Args:
            xai_api_key: xAI API key, defaults to XAI_API_KEY env var
            model: Model name, e.g. "grok-2"
            temperature: Sampling temperature
            model_kwargs: Additional kwargs passed to ChatXAI
        
        Raises:
            ValueError: If no API key is provided or found in env vars
        """
        if xai_api_key is None:
            xai_api_key = os.getenv("XAI_API_KEY", "")

        if not xai_api_key:
            raise ValueError(
                "No xAI API key found. Provide xai_api_key directly "
                "or set XAI_API_KEY in your environment or .env file."
            )

        self._chat = ChatXAI(
            xai_api_key=xai_api_key,
            model=model,
            temperature=temperature,
            **model_kwargs
        )
    
    def generate_messages(self, messages: list[tuple[str, str]]) -> str:
        """
        Send messages to xAI and get a text response.
        
        Args:
            messages: List of (role, content) tuples
            
        Returns:
            Text response from the model
        """
        response = self._chat.invoke(messages)
        return response.content


if __name__ == "__main__":
    print("=== XAI Model Test ===")
    xai_model = XAIChatModel(
        model="grok-2",
        temperature=0.2
    )
    msgs = [
        ("system", "You are a helpful translator. Respond only in French."),
        ("human", "I love programming.")
    ]
    response = xai_model.generate_messages(msgs)
    print("Response:", response)
    print("=== End of Test ===")