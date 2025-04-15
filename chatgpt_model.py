import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from role_models import CompleteChatModel

load_dotenv()

class ChatGPTChatModel(CompleteChatModel):
    """
    OpenAI's ChatGPT implementation using LangChain.
    Inherits all JSON generation methods from CompleteChatModel.
    """

    def __init__(
        self, 
        openai_api_key: str = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        **model_kwargs
    ):
        """
        Initialize the ChatGPT model.
        
        Args:
            openai_api_key: OpenAI API key, defaults to OPENAI_API_KEY env var
            model: Model name, e.g. "gpt-3.5-turbo", "o1"
            temperature: Sampling temperature (1.0 recommended for o-series)
            model_kwargs: Additional kwargs passed to ChatOpenAI
        
        Raises:
            ValueError: If no API key is provided or found in env vars
        """
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY", "")

        if not openai_api_key:
            raise ValueError(
                "No OpenAI API key found. Provide openai_api_key directly "
                "or set OPENAI_API_KEY in your environment or .env file."
            )

        self._chat = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model,
            temperature=temperature,
            **model_kwargs
        )
    
    def generate_messages(self, messages: list[tuple[str, str]]) -> str:
        """
        Send messages to ChatGPT and get a text response.
        
        Args:
            messages: List of (role, content) tuples
            
        Returns:
            Text response from the model
        """
        response = self._chat.invoke(messages)
        return response.content


if __name__ == "__main__":
    print("=== ChatGPT Model Test ===")
    gpt_model = ChatGPTChatModel(model="o1")
    msgs = [
        ("system", "You are a helpful translator. Respond only in French."),
        ("human", "I love programming.")
    ]
    response = gpt_model.generate_messages(msgs)
    print("Response:", response)
    print("=== End of Test ===")