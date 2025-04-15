from base import BaseChatModel, BotAInterface, BotBInterface, JudgeInterface
from base import BotAResponse, FakeQuestion, BotBResponse, JudgeResponse, FinalJudge

class CompleteChatModel(BaseChatModel, BotAInterface, BotBInterface, JudgeInterface):
    """
    A complete implementation that can fulfill all roles.
    This class implements all the role-specific JSON generation methods using the base templates.
    """
    
    def generate_messages_json_botA(self, messages: list[tuple[str, str]]) -> str:
        """
        For Chat Bot A to generate messages in JSON format.
        
        Args:
            messages: List of (role, content) tuples
            
        Returns:
            JSON string with Bot A's response
        """
        format_instructions = """
        When presenting your quiz questions, respond in JSON format:
        For the question, you need to include:
           - "evaluation_Bot_B": your brief evaluation of Bot B's answer after they respond. If Bot B hasn't responded yet, leave this as an empty string.
           - "question": the main quiz question text.
           - "tactic": a description of the tactic you're using to challenge Bot B.
           - "correction_context": the correction or additional context to be revealed after Bot B answers (if none, use an empty string).
        """
        return self.generate_json_output(messages, BotAResponse, format_instructions)
    
    def generate_messages_json_botA_fake(self, messages: list[tuple[str, str]]) -> str:
        """
        For Chat Bot A to generate fake messages in JSON format.
        
        Args:
            messages: List of (role, content) tuples
            
        Returns:
            JSON string with Bot A's fake question
        """
        format_instructions = """
        You need to ask a fake question or a non-existing question based on the given domain (a 'fake' phenomenon, theory, or experiment),
        follow these steps and respond in JSON format:

        1. "evaluation_Bot_B": your brief evaluation of Bot B's answer after they respond. If Bot B hasn't responded yet, leave this as an empty string.
        2. Generate a question that is intentionally fabricated, blending plausible details with obviously fake elements.
        3. Provide the question text under "question".
        4. IMPORTANT: Do NOT reveal that the question is fake in your output. The "tactic" field must be empty.
        5. IMPORTANT: Do NOT reveal that the question is fake in your output. The "explanation" field must be empty.
        6. Your output MUST be a valid JSON object with the keys "evaluation_Bot_B", "question", "tactic", and "explanation".
        """
        return self.generate_json_output(messages, FakeQuestion, format_instructions)
    
    def generate_messages_json_botB(self, messages: list[tuple[str, str]]) -> str:
        """
        For Chat Bot B to generate messages in JSON format.
        
        Args:
            messages: List of (role, content) tuples
            
        Returns:
            JSON string with Bot B's response
        """
        format_instructions = """
        When analyzing a question, you must follow these steps and respond in JSON format:

        1. Begin with a brief analysis message indicating you're processing the question.
        2. Determine a "difficulty_rating" based on the question's complexity (e.g., "7/10").
        3. Outline your "reasoning_steps" as a numbered list. Each step should explain a part of your thought process.
        4. Provide your "final_answer" with a clear, concise response based on your analysis.
        5. Your output MUST be a valid JSON object with exactly these keys: "difficulty_rating", "reasoning_steps", and "final_answer".
        """
        return self.generate_json_output(messages, BotBResponse, format_instructions)
    
    def generate_messages_json_judge(self, messages: list[tuple[str, str]]) -> str:
        """
        Generate messages for the Judge role in JSON format.
        
        Args:
            messages: List of (role, content) tuples
            
        Returns:
            JSON string with Judge's evaluation
        """
        format_instructions = """
        When evaluating the Q&A session, answer the following judge prompts.
        For each prompt, provide your answer. Your output MUST be a valid JSON object with exactly one key: "responses".
        The value of "responses" must be an object where each key is one of the judge prompt questions and each corresponding value is your answer.
        Here are the judge prompts you need to answer:

        1) "Is Bot A's question sufficiently challenging for this domain (or is it too trivial/specialized)?"
        2) "Rate the question's quality from 1 to 10 (1=too trivial, 10=excellent, broad and conceptual)."
        3) "Is Bot B's answer correct and well-explained? Rate from 1 to 10 (1=not correct/nonsense, 10=fully correct and thorough)."
        4) "Provide a brief critique (separate your critique for Bot A's question and Bot B's answer)."
        5) "Give your own short answer to Bot A's question."
        6) "Do you think Bot A is taking the right approach in stumping Bot B? Rate 1-10 and explain."
        """
        return self.generate_json_output(messages, JudgeResponse, format_instructions)
    
    def generate_messages_json_final_judge(self, messages: list[tuple[str, str]]) -> str:
        """
        Generate messages for the Final Judge role in JSON format.
        
        Args:
            messages: List of (role, content) tuples
            
        Returns:
            JSON string with Final Judge's evaluation
        """
        format_instructions = """
        You are a final round judge evaluating the entire Q&A session between Bot A and Bot B.
        Your output MUST be a valid JSON object with exactly one key: "responses".
        The value of "responses" must be an object where each key is one of the judge prompt questions and each corresponding value is your answer.
        Here are the judge prompts you need to answer:

        1) "Did Bot A succeed in stumping Bot B?"
        2) "Was Bot B's performance strong overall?"
        3) "How was the overall question quality (1-10) across all rounds?"
        4) "Any final remarks?"
        """
        return self.generate_json_output(messages, FinalJudge, format_instructions)