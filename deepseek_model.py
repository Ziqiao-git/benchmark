import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from role_models import CompleteChatModel

load_dotenv()

class DeepSeekChatModel(CompleteChatModel):
    """
    DeepSeek's implementation using LangChain's ChatOpenAI with custom API base.
    Inherits all JSON generation methods from CompleteChatModel.
    """

    def __init__(
        self,
        deepseek_api_key: str = None,
        model: str = "deepseek-reasoner",
        temperature: float = 0.2,
        **model_kwargs
    ):
        """
        Initialize the DeepSeek model.
        
        Args:
            deepseek_api_key: DeepSeek API key, defaults to DEEPSEEK_API_KEY env var
            model: Model name, e.g. "deepseek-reasoner"
            temperature: Sampling temperature
            model_kwargs: Additional kwargs passed to ChatOpenAI
        
        Raises:
            ValueError: If no API key is provided or found in env vars
        """
        if deepseek_api_key is None:
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")

        if not deepseek_api_key:
            raise ValueError(
                "No DeepSeek API key found. Provide deepseek_api_key directly "
                "or set DEEPSEEK_API_KEY in your environment or .env file."
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
        Send messages to DeepSeek and get a text response.
        
        Args:
            messages: List of (role, content) tuples
            
        Returns:
            Text response from the model
        """
        response = self._chat.invoke(messages)
        return response.content


if __name__ == "__main__":
    print("=== DeepSeek Model Test ===")
    deepseek_model = DeepSeekChatModel(
        model="deepseek-reasoner",
        temperature=0.2
    )
    msgs = [
        ("system", "You are a helpful translator. Respond only in French."),
        ("human", "I love programming.")
    ]
    response = deepseek_model.generate_messages(msgs)
    print("Response:", response)
    print("=== End of Test ===")