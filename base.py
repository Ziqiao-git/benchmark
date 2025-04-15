from abc import ABC, abstractmethod
import json
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from typing import TypeVar, Type, Dict, Any, Optional

# Define a generic type for Pydantic models
T = TypeVar('T', bound=BaseModel)

class ChatModelInterface(ABC):
    """A standard interface for any chat model."""

    @abstractmethod
    def generate_messages(self, messages: list[tuple[str, str]]) -> str:
        """Send a prompt to the model and return its text response."""
        pass

class BaseChatModel(ChatModelInterface):
    """Base implementation with common JSON generation patterns."""
    
    def _add_json_instructions(self, messages: list[tuple[str, str]], instructions: str) -> list[tuple[str, str]]:
        """
        Add JSON formatting instructions to the system prompt.
        
        Args:
            messages: List of (role, content) tuples
            instructions: JSON formatting instructions to add
            
        Returns:
            Modified messages list with instructions added to system prompt
        """
        messages_copy = messages.copy()
        if messages_copy and messages_copy[0][0] == "system":
            combined_system = messages_copy[0][1] + "\n" + instructions
            messages_copy[0] = ("system", combined_system)
        else:
            messages_copy.insert(0, ("system", instructions))
        return messages_copy
    
    def _generate_with_parser(self, 
                             messages: list[tuple[str, str]], 
                             pydantic_model: Type[T], 
                             max_retries: int = 5) -> str:
        """
        Generate a response and parse it using a Pydantic model.
        
        Args:
            messages: List of (role, content) tuples
            pydantic_model: Pydantic model class to parse the response
            max_retries: Maximum number of generation attempts
            
        Returns:
            JSON string of the parsed response
            
        Raises:
            ValueError: If parsing fails after max_retries attempts
        """
        output_parser = PydanticOutputParser(pydantic_object=pydantic_model)
        
        for attempt in range(1, max_retries + 1):
            response_text = self.generate_messages(messages)
            try:
                parsed = output_parser.parse(response_text)
                return json.dumps(parsed.dict(), ensure_ascii=False, indent=2)
            except Exception as e:
                if attempt == max_retries:
                    raise ValueError(
                        f"Could not parse response after {max_retries} attempts.\n"
                        f"Last response: {response_text}\n"
                        f"Error: {str(e)}"
                    )

    def generate_json_output(self, 
                            messages: list[tuple[str, str]], 
                            pydantic_model: Type[T], 
                            format_instructions: str,
                            max_retries: int = 5) -> str:
        """
        Generate JSON output using a specified Pydantic model and format instructions.
        
        Args:
            messages: List of (role, content) tuples
            pydantic_model: Pydantic model to use for output parsing
            format_instructions: Instructions for formatting the JSON output
            max_retries: Maximum number of generation attempts
            
        Returns:
            JSON string of the parsed response
        """
        output_parser = PydanticOutputParser(pydantic_object=pydantic_model)
        parser_instructions = output_parser.get_format_instructions()
        
        json_instructions = f"""
        Your response must start with {{ and end with }} with no additional text.
        {format_instructions}
        
        Format your answer using these instructions:
        {parser_instructions}
        
        Your response must start with {{ and end with }} with no additional text.
        """
        
        modified_messages = self._add_json_instructions(messages, json_instructions)
        return self._generate_with_parser(modified_messages, pydantic_model, max_retries)

# Role-specific model definitions
class BotAInterface:
    """Interface for models that can act as Bot A."""
    
    @abstractmethod
    def generate_messages_json_botA(self, messages: list[tuple[str, str]]) -> str:
        """Generate messages for Bot A role."""
        pass
    
    @abstractmethod
    def generate_messages_json_botA_fake(self, messages: list[tuple[str, str]]) -> str:
        """Generate fake messages for Bot A role."""
        pass

class BotBInterface:
    """Interface for models that can act as Bot B."""
    
    @abstractmethod
    def generate_messages_json_botB(self, messages: list[tuple[str, str]]) -> str:
        """Generate messages for Bot B role."""
        pass

class JudgeInterface:
    """Interface for models that can act as a Judge."""
    
    @abstractmethod
    def generate_messages_json_judge(self, messages: list[tuple[str, str]]) -> str:
        """Generate messages for Judge role."""
        pass
    
    @abstractmethod
    def generate_messages_json_final_judge(self, messages: list[tuple[str, str]]) -> str:
        """Generate messages for Final Judge role."""
        pass

# Pydantic model definitions for each role
class BotAResponse(BaseModel):
    """Pydantic model for Bot A responses."""
    evaluation_Bot_B: str
    question: str
    tactic: str
    correction_context: str

class FakeQuestion(BaseModel):
    """Pydantic model for fake questions from Bot A."""
    evaluation_Bot_B: str
    question: str
    tactic: str
    explanation: str

class BotBResponse(BaseModel):
    """Pydantic model for Bot B responses."""
    difficulty_rating: str
    reasoning_steps: Dict[str, str]
    final_answer: str

class JudgeResponse(BaseModel):
    """Pydantic model for Judge responses."""
    responses: Dict[str, str]

class FinalJudge(BaseModel):
    """Pydantic model for Final Judge responses."""
    responses: Dict[str, str]