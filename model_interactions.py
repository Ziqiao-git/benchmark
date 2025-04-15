# model_interactions.py
from model_handler import get_chat_model
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional

class ParticipantInterface(ABC):
    """Base interface for all participants in a debate or evaluation."""
    
    @abstractmethod
    def generate_response(self, context: dict) -> str:
        """Generate a response based on provided context."""
        pass

class ModelParticipant(ParticipantInterface):
    """Wrapper for LLM models to participate in debates/evaluations."""
    
    def __init__(self, model_id: str, role: str = "participant"):
        """
        Initialize a model participant.
        
        Args:
            model_id: ID of model in config
            role: Role of this participant (debater, judge, etc.)
        """
        self.model = get_chat_model(model_id)
        self.model_id = model_id
        self.role = role
        self.history = []
    
    def generate_response(self, context: dict) -> str:
        """Generate a response from the model based on context."""
        # Format messages based on context
        messages = self._format_messages(context)
        
        # Get response from model
        response = self.model.generate_messages(messages)
        
        # Update history
        self.history.append({"context": context, "response": response})
        
        return response
    
    def _format_messages(self, context: dict) -> List[Tuple[str, str]]:
        """Format context into messages for the model."""
        messages = []
        
        # Add system message if available
        if "system_prompt" in context:
            messages.append(("system", context["system_prompt"]))
        
        # Add conversation history
        if "history" in context:
            for entry in context["history"]:
                if "user" in entry:
                    messages.append(("user", entry["user"]))
                if "assistant" in entry:
                    messages.append(("assistant", entry["assistant"]))
        
        # Add current query/input
        if "input" in context:
            messages.append(("user", context["input"]))
            
        return messages


class Debate:
    """Manages a structured debate between models."""
    
    def __init__(self, topic: str, participants: List[ModelParticipant], 
                 judge: Optional[ModelParticipant] = None, rounds: int = 3):
        """
        Initialize a debate session.
        
        Args:
            topic: The debate topic
            participants: List of model participants
            judge: Optional judge model participant
            rounds: Number of debate rounds
        """
        self.topic = topic
        self.participants = participants
        self.judge = judge
        self.rounds = rounds
        self.transcript = []
        
    def run(self) -> Dict[str, Any]:
        """Run the full debate and return results."""
        # Initialize the debate
        system_prompt = f"You are participating in a structured debate on the topic: {self.topic}"
        
        # Run debate rounds
        for round_num in range(1, self.rounds+1):
            round_responses = {}
            round_prompt = f"This is round {round_num} of the debate on {self.topic}. "
            
            if round_num == 1:
                round_prompt += "Please present your initial position and arguments."
            else:
                round_prompt += "Respond to the previous arguments and further develop your position."
            
            # Get response from each participant
            for participant in self.participants:
                context = {
                    "system_prompt": system_prompt,
                    "input": round_prompt,
                    "history": self.transcript,
                    "round": round_num
                }
                
                response = participant.generate_response(context)
                round_responses[participant.model_id] = response
                
                # Add to transcript
                self.transcript.append({
                    "round": round_num,
                    "participant": participant.model_id,
                    "response": response
                })
        
        # Judge the debate if a judge is provided
        results = {"transcript": self.transcript}
        if self.judge:
            judge_context = {
                "system_prompt": f"You are judging a debate on the topic: {self.topic}. "
                                 f"Evaluate the arguments of all participants and determine a winner.",
                "input": "Based on the debate transcript, evaluate the arguments and determine a winner.",
                "history": [{"user": f"Debate transcript: {self.transcript}"}]
            }
            
            judgment = self.judge.generate_response(judge_context)
            results["judgment"] = judgment
        
        return results


class Evaluation:
    """Manages evaluations and voting between model responses."""
    
    def __init__(self, judges: List[ModelParticipant], criteria: Optional[List[str]] = None):
        """
        Initialize an evaluation session.
        
        Args:
            judges: List of judge model participants
            criteria: Optional list of evaluation criteria
        """
        self.judges = judges
        self.criteria = criteria or ["accuracy", "coherence", "helpfulness"]
    
    def evaluate_pair(self, prompt: str, response_a: str, response_b: str, 
                      model_a_id: str, model_b_id: str) -> Dict[str, Any]:
        """
        Have judges evaluate a pair of responses.
        
        Args:
            prompt: The original prompt given to models
            response_a: Response from model A
            response_b: Response from model B
            model_a_id: Identifier for model A
            model_b_id: Identifier for model B
            
        Returns:
            Evaluation results
        """
        results = {"votes": {}, "explanations": {}}
        
        for judge in self.judges:
            # Create evaluation context
            context = {
                "system_prompt": (
                    f"You are evaluating two AI responses to the same prompt. "
                    f"Judge which response better satisfies these criteria: {', '.join(self.criteria)}. "
                    f"Choose only one winner from the two responses."
                ),
                "input": (
                    f"Original prompt: {prompt}\n\n"
                    f"Response A ({model_a_id}):\n{response_a}\n\n"
                    f"Response B ({model_b_id}):\n{response_b}\n\n"
                    f"Which response is better overall? Answer with either 'A' or 'B', "
                    f"followed by a brief explanation of your decision."
                )
            }
            
            # Get judgment
            judgment = judge.generate_response(context)
            
            # Parse vote (first character should be A or B)
            vote = None
            if judgment.strip().startswith('A'):
                vote = model_a_id
            elif judgment.strip().startswith('B'):
                vote = model_b_id
            
            # Store results
            results["votes"][judge.model_id] = vote
            results["explanations"][judge.model_id] = judgment
        
        # Count votes
        vote_counts = {model_a_id: 0, model_b_id: 0}
        for vote in results["votes"].values():
            if vote:
                vote_counts[vote] += 1
        
        # Determine winner
        if vote_counts[model_a_id] > vote_counts[model_b_id]:
            results["winner"] = model_a_id
        elif vote_counts[model_b_id] > vote_counts[model_a_id]:
            results["winner"] = model_b_id
        else:
            results["winner"] = "tie"
        
        results["vote_counts"] = vote_counts
        return results