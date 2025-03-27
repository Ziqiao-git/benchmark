# chat_models/deepseek_model.py
import json
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from base import ChatModelInterface
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser


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

    def generate_messages_json_botA(self, messages: list[tuple[str, str]]) -> str:
        """
        For Chat Bot A to generated messages
        Make responds return in Json format:
        :param messages: a list of (role, content) tuples, e.g. [("system", "..."), ("human", "...")]
        :return: model's final text response in json file.
        """

        # Define a Pydantic model representing the expected JSON output.
        class BotAResponse(BaseModel):
            # question: list[Question]
            question: str
            tactic: str
            correction_context: str

        output_parser = PydanticOutputParser(pydantic_object=BotAResponse)
        format_instructions = output_parser.get_format_instructions()

        json_instructions = f"""
        Your response must start with {{ and end with }} with no additional text.
        When presenting your quiz questions, respond in JSON format:
        For the question, you need to include:
           - "question_text": the main quiz question text.
           - "tactic": a description of the tactic you’re using to challenge Bot B.
           - "correction_context": the correction or additional context to be revealed after Bot B answers (if none, use an empty string).

        Format your answer using these instructions:
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

    def generate_messages_json_botB(self, messages: list[tuple[str, str]]) -> str:
        """
        For Chat Bot B to generated messages
        Make responds return in Json format:
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
                Your response must start with {{ and end with }} with no additional text
                When analyzing a question, you must follow these steps and respond in JSON format:

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

    def generate_messages_json_judge(self, messages: list[tuple[str, str]]) -> str:
        """
        Generated messages prompts that use for judge model in normal role.
        Make responds return in Json format:
        :param messages: a list of (role, content) tuples, e.g. [("system", "..."), ("human", "...")]
        :return: model's final text response in json file.
        """

        class JudgeResponse(BaseModel):
            responses: dict[str, str]

        output_parser = PydanticOutputParser(pydantic_object=JudgeResponse)
        format_instructions = output_parser.get_format_instructions()
        json_instructions = f"""
        When evaluating the Q&A session, answer the following judge prompts.
        Your response must start with {{ and end with }} with no additional text.
        For each prompt, provide your answer. Your output MUST be a valid JSON object with exactly one key: "responses".
        The value of "responses" must be an object where each key is one of the judge prompt questions and each corresponding value is your answer.
        Here are the judge prompts you need to answer:

        1) "Is Bot A's question sufficiently challenging for this domain (or is it too trivial/specialized)?"
        2) "Rate the question's quality from 1 to 10 (1=too trivial, 10=excellent, broad and conceptual)."
        3) "Is Bot B's answer correct and well-explained? Rate from 1 to 10 (1=not correct/nonsense, 10=fully correct and thorough)."
        4) "Provide a brief critique (separate your critique for Bot A's question and Bot B's answer)."
        5) "Give your own short answer to Bot A's question."
        6) "Do you think Bot A is taking the right approach in stumping Bot B? Rate 1-10 and explain."

        Format your answer using these instructions:
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

    def generate_messages_json_final_judge(self, messages: list[tuple[str, str]]) -> str:
        """
        Generated messages prompts that use for judge model in final role.
        Make responds return in Json format:
        :param messages: a list of (role, content) tuples, e.g. [("system", "..."), ("human", "...")]
        :return: model's final text response in json file.
        """

        class FinalJudge(BaseModel):
            responses: dict[str, str]

        output_parser = PydanticOutputParser(pydantic_object=FinalJudge)
        format_instructions = output_parser.get_format_instructions()
        json_instructions = f"""
        You are a final round judge evaluating the entire Q&A session between Bot A and Bot B.
        Your response must start with {{ and end with }} with no additional text.
        Your output MUST be a valid JSON object with exactly one key: "responses".
        The value of "responses" must be an object where each key is one of the judge prompt questions and each corresponding value is your answer.
        Here are the judge prompts you need to answer:

        1) "Did Bot A succeed in stumping Bot B?"
        2) "Was Bot B's performance strong overall?"
        3) "How was the overall question quality (1-10) across all rounds?"
        4) "Any final remarks?"

        Format your answer using these instructions:
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
