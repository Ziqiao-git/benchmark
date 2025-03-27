# chat_models/claude_model.py
import json
from langchain_anthropic import ChatAnthropic
from base import ChatModelInterface
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser

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

    def generate_messages_json(self, messages: list[tuple[str, str]]) -> str:
        """
        Make them return in Json format:
        :param messages: a list of (role, content) tuples, e.g. [("system", "..."), ("human", "...")]
        :return: model's final text response in json file.
        """

        # Define a Pydantic model representing the expected JSON output.
        class BotBResponse(BaseModel):
            difficulty_rating: str
            reasoning_steps: dict[str, str]
            final_answer: str

        output_parser = PydanticOutputParser(pydantic_object=BotBResponse)
        format_instructions = output_parser.get_format_instructions()

        json_instructions = f"""
        You are Bot B. When analyzing a question, you must follow these steps and respond in JSON format:

        1. Begin with a brief analysis message indicating you're processing the question.
        2. Determine a "difficulty_rating" based on the question's complexity (e.g., "7/10").
        3. Outline your "reasoning_steps" as a numbered list. Each step should explain a part of your thought process.
        4. Provide your "final_answer" with a clear, concise response based on your analysis.
        5. Your output MUST be a valid JSON object with exactly these keys: "difficulty_rating", "reasoning_steps", and "final_answer".

        Format your answer following these instructions:
        {format_instructions}

        Your response must start with {{ and end with }} with no additional text.
        """
        if messages and messages[0][0] == "system":
            combined_system = messages[0][1] + "\n" + json_instructions
            messages[0] = ("system", combined_system)
        else:
            messages.insert(0, ("system", json_instructions))

        # Can adjust the max retries. It will return error after max tries.
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            response_text = self.generate_messages(messages)
            try:
                parsed = output_parser.parse(response_text)
                return json.dumps(parsed.dict(), ensure_ascii=False, indent=2)
            except Exception:
                if attempt == max_retries:
                    raise ValueError(
                        f"Could not parse response after {max_retries} attempts.\nLast response: {response_text}"
                    )

# Quick test
if __name__ == "__main__":
    print("=== Claude Model Test ===")

    # We'll assume CLAUDE_API_KEY is set in your environment / .env file.
    claude_model = ClaudeChatModel(
        model="claude-3-7-sonnet-20250219",
        temperature=0
    )

    msg = [
        ("system", "You are a helpful translator. Respond only in French. Return in json format"),
        ("human", "I love programming.")
    ]
    response = claude_model.generate_messages_json(msg)
    # response = claude_model.generate_messages(msg)
    print("Response:", response)
    print("=== End of Test ===")
