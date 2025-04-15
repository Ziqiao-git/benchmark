import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from role_models import CompleteChatModel

load_dotenv()

class ClaudeChatModel(CompleteChatModel):
    """
    Anthropic's Claude implementation using LangChain.
    Inherits all JSON generation methods from CompleteChatModel.
    """
    
    def __init__(
        self,
        anthropic_api_key: str = None,
        model: str = "claude-2",
        temperature: float = 0.0,
        **model_kwargs
    ):
        """
        Initialize the Claude model.
        
        Args:
            anthropic_api_key: Anthropic API key, defaults to CLAUDE_API_KEY env var
            model: Model name, e.g. "claude-2", "claude-3-7-sonnet-20250219"
            temperature: Sampling temperature (0.0 for deterministic outputs)
            model_kwargs: Additional kwargs passed to ChatAnthropic
            
        Raises:
            ValueError: If no API key is provided or found in env vars
        """
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
        Send messages to Claude and get a text response.
        
        Args:
            messages: List of (role, content) tuples
            
        Returns:
            Text response from the model
        """
        response = self._chat.invoke(messages)
        return response.content


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