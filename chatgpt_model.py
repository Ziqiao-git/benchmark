# chat_models/chatgpt_model.py
from langchain.chat_models import ChatOpenAI
# chatgpt_model.py
import json
import os
from dotenv import load_dotenv
from base import ChatModelInterface
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser

load_dotenv()

class ChatGPTChatModel(ChatModelInterface):
    def __init__(
        self, 
        openai_api_key: str = None,
        model: str = "gpt-3.5-turbo",
        **model_kwargs
    ):

        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY", "")

        if not openai_api_key:
            raise ValueError("No OpenAI API key found. ...")

        self._chat = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model,
            temperature=1.0, #always 1.0 for o-series
            **model_kwargs
        )
    
    def generate_messages(self, messages: list[tuple[str, str]]) -> str:
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
            evaluation_Bot_B: str
            question: str
            tactic: str
            correction_context: str

        output_parser = PydanticOutputParser(pydantic_object=BotAResponse)
        format_instructions = output_parser.get_format_instructions()

        json_instructions = f"""
        Your response must start with {{ and end with }} with no additional text.
        When presenting your quiz questions, respond in JSON format:
        For the question, you need to include:
           - "evaluation": your brief evaluation of Bot B's answer after they respond. If Bot B hasn't responded yet, leave this as an empty string.
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
        max_retries = 5
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

    def generate_messages_json_botA_fake(self, messages: list[tuple[str, str]]) -> str:
        """
            For Chat Bot A to generated **fake** messages
            Make responds return in Json format:
            :param messages: a list of (role, content) tuples, e.g. [("system", "..."), ("human", "...")]
            :return: model's final text response in json file.

        """

        class FakeQuestion(BaseModel):
            evaluation_Bot_B: str
            question: str
            tactic: str
            explanation: str

        output_parser = PydanticOutputParser(pydantic_object=FakeQuestion)
        format_instructions = output_parser.get_format_instructions()
        json_instructions = f"""
            You need to ask a fake question or a non-existing question based on the given domain (a ‘fake’ phenomenon, theory, or experiment),
            follow these steps and respond in JSON format:

            1. - "evaluation": your brief evaluation of Bot B's answer after they respond. If Bot B hasn't responded yet, leave this as an empty string.
            2. Generate a question that is intentionally fabricated, blending plausible details with obviously fake elements.
            3. Provide the question text under "question".
            4. IMPORTANT: Do NOT reveal that the question is fake in your output. The "tactic" field must be empty.
            5. IMPORTANT: Do NOT reveal that the question is fake in your output. The "explanation" field must be empty.
            6. Your output MUST be a valid JSON object with an object with the keys "question", "tactic", and "explanation".

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
        max_retries = 5
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
        max_retries = 5
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
        max_retries = 5
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
        max_retries = 5
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
    print("=== ChatGPT Model Test ===")
    gpt_model = ChatGPTChatModel(model="o1")
    msgs = [
        ("system", "You are a helpful translator. Respond only in French."),
        ("human", "I love programming.")
    ]
    response = gpt_model.generate_messages(msgs)
    print("Response:", response)
    print("=== End of Test ===")
