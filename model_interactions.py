# model_interactions.py
from model_handler import get_chat_model
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate


class ParticipantInterface(ABC):
    """Base interface for all participants in a debate or evaluation."""
    
    @abstractmethod
    def generate_response(self, context: dict) -> str:
        """Generate a response based on provided context."""
        pass

class ModelParticipant(ParticipantInterface):
    """Wrapper for LLM models to participate in debates/evaluations."""
    
    def __init__(self, model_id: str, role: str = "participant", **kwargs):
        """
        Initialize a model participant.
        
        Args:
            model_id: ID of model in config
            role: Role of this participant (debater, judge, etc.)
            **kwargs: Additional arguments to pass to get_chat_model
        """
        self.model = get_chat_model(model_id, **kwargs)
        self.model_id = kwargs.get('custom_model_id', model_id)  # Use custom_model_id if provided
        self.role = role
        self.history = []
    
    def generate_response(self, context: dict) -> str:
        """Generate a response from the model based on context."""
        # Format messages based on context
        messages = self._format_messages(context)
        print(messages)
        
        # Get response from model
        response = self.model.generate_messages(messages)
        
        # Update history
        self.history.append({"context": context, "response": response})
        
        return response
    
    def _format_messages(self, context: dict) -> List[tuple[str, str]]:
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

    def __init__(
        self,
        topic: str,
        participants: List[ModelParticipant],
        judges: List[ModelParticipant] = None,
        rounds: int = 3,
        detailed_instructions: str = None
    ):
        """
        Initialize a debate session.

        Args:
            topic: The debate topic
            participants: List of model participants
            judges: Optional judge model participant
            rounds: Number of debate rounds
            detailed_instructions: Optional extra instructions
        """
        self.topic = topic
        self.participants = participants
        self.rounds = rounds
        self.transcript = []
        self.detailed_instructions = detailed_instructions

    def _get_history_for(self, participant: ModelParticipant, round_num: int) -> List[Dict]:
        """
        Retrieve the conversation history for this participant, showing
        *only the messages that participant has produced* in all rounds
        up to (and including) 'round_num'.

        This means the participant never sees the other model’s questions
        (nor the other model’s answers) in 'history'. They only see
        what they themselves wrote in previous steps.
        """
        filtered_history = []
        for entry in self.transcript:
            if entry["round"] <= round_num and entry["participant"] == participant.model_id:
                filtered_history.append(entry)
        return filtered_history

    def _format_history(self, participant: ModelParticipant, history_entries: List[Dict]) -> List[Dict]:
        """
        Convert the raw list of transcript entries into the format used for
        the 'history' in the context. Typically, we produce a list of
        dictionaries with either {"user": "..."} or {"assistant": "..."} according to the model's role.
        that the model can consume in `_format_messages()`.
        """
        formatted = []
        for entry in history_entries:
            same_participant = entry["participant"] == participant.model_id
            if same_participant and entry["role"] == "challenger":
                # The participant's own lines => assistant but we need to add a user message to the history (deepseek need to see the user message first)
                formatted.append({"user": "Please strictly follow the giving tone and setting, propose your question now"})
                formatted.append({"assistant": entry["response"]})
            elif same_participant and entry["role"] == "responder":
                # The other model's lines => user
                formatted.append({"assistant": entry["response"]})
            else:
                # The other model's lines => user
                formatted.append({"user": entry["response"]})
        return formatted

    def run(self) -> Dict[str, Any]:
        """Run the full debate and return results."""
        if len(self.participants) != 2:
            raise ValueError("Debate must have exactly two participants")

        participant_a, participant_b = self.participants

        if self.detailed_instructions is None:
            self.detailed_instructions = "Remember to be creative and think outside the box."

        challenge_prompt = {
            "system_prompt": (
                f"You are a challenger on the topic of {self.topic}. "
                f"Ask a difficult question that will test your opponent's knowledge. "
                f"Here are some detailed instructions: {self.detailed_instructions}"
            ),
            "input": None,   # Will fill in each time
            "history": None, # Will fill in each time
            "round": None
        }

        response_prompt = {
            "system_prompt": (
                f"You are answering a challenging question on the topic of {self.topic}."
            ),
            "input": None,   # Will fill in each time
            "history": None, # Will fill in each time
            "round": None
        }

        for round_num in range(1, self.rounds + 1):

            # =========================
            # 1) Participant A asks a question
            # =========================
            context_a = challenge_prompt.copy()
            # HISTORY: only A's own prior messages
            context_a["history"] = self._format_history(
                participant_a,
                self._get_history_for(participant_a, round_num)
            )
            # PROMPT
            context_a["input"] = (
                f"Create a challenging question about {self.topic} that "
                "will be difficult for your opponent to answer correctly. "
                "Use any past context you have (only your own) to refine it."
            )
            context_a["round"] = round_num

            challenge_a = participant_a.generate_response(context_a)
            self.transcript.append({
                "round": round_num,
                "role": "challenger",
                "participant": participant_a.model_id,
                "response": challenge_a
            })

            # =========================
            # 2) Participant B answers A's question
            # =========================
            context_b = response_prompt.copy()
            # HISTORY: only B's own prior messages
            context_b["history"] = self._format_history(
                participant_b,
                self._get_history_for(participant_b, round_num)
            )
            # The question from participant A is *only* in the input
            context_b["input"] = f"Answer the following question about {self.topic}: {challenge_a}"
            context_b["round"] = round_num

            response_b = participant_b.generate_response(context_b)
            self.transcript.append({
                "round": round_num,
                "role": "responder",
                "participant": participant_b.model_id,
                "response": response_b
            })

            # =========================
            # 3) Participant B asks a question
            # =========================
            context_b = challenge_prompt.copy()
            context_b["history"] = self._format_history(
                participant_b,
                self._get_history_for(participant_b, round_num)
            )
            context_b["input"] = (
                f"Create a challenging question about {self.topic} that "
                "will be difficult for your opponent to answer correctly. "
                "Use any past context you have (only your own) to refine it."
            )
            context_b["round"] = round_num

            challenge_b = participant_b.generate_response(context_b)
            self.transcript.append({
                "round": round_num,
                "role": "challenger",
                "participant": participant_b.model_id,
                "response": challenge_b
            })

            # =========================
            # 4) Participant A answers B's question
            # =========================
            context_a = response_prompt.copy()
            context_a["history"] = self._format_history(
                participant_a,
                self._get_history_for(participant_a, round_num)
            )
            # The question from participant B is only in the input
            context_a["input"] = f"Answer the following question about {self.topic}: {challenge_b}"
            context_a["round"] = round_num

            response_a = participant_a.generate_response(context_a)
            self.transcript.append({
                "round": round_num,
                "role": "responder",
                "participant": participant_a.model_id,
                "response": response_a
            })

        # At the end, return the full transcript
        return {"transcript": self.transcript}


class JudgmentCriteria(BaseModel):
    """Evaluation criteria for a round"""
    # question_quality: str = Field(..., description="Assessment of the quality of both models' questions")
    # answer_quality: str = Field(..., description="Assessment of the quality of both models' answers")
    # reasoning: str = Field(..., description="Explanation for why one model performed better than the other")
    winner: Literal["A", "B"] = Field(..., description="Which model performed better in answering the rival proposed question under the response criteria (A or B)")
    better_question: Literal["A", "B"] = Field(..., description="Which model asked the better question under the question criteria (A or B)")

class FinalAssessment(BaseModel):
    """Final assessment of the entire battle"""
    # overall_performance: str = Field(..., description="Assessment of both models' overall performance")
    # history_usage: str = Field(..., description="Assessment of how well each model used conversation history to attack the opponent")
    better_history_user: Literal["A", "B"] = Field(..., description="Which model better used history to attack (A or B)")
    final_winner: Literal["A", "B"] = Field(..., description="Overall battle winner (A or B)")
    # reasoning: str = Field(..., description="Explanation for the final judgment")

class Evaluation:
    """Manages evaluations and voting between model responses in a debate."""
    
    def __init__(self, judges: List[ModelParticipant], transcript: List[Dict], 
                 model_a_id: str, model_b_id: str, topic: str = None, response_criteria: Optional[List[str]] = None, question_criteria: Optional[List[str]] = None, detailed_instructions: str = None):
        """
        Initialize an evaluation session.
        
        Args:
            judges: List of judge model participants
            transcript: The debate transcript with questions and answers
            model_a_id: Identifier for model A
            model_b_id: Identifier for model B
            response_criteria: Optional list of evaluation criteria
            detailed_instructions: instructions that judges should 
        """
        self.judges = judges
        self.topic = topic
        self.transcript = transcript
        self.model_a_id = model_a_id
        self.model_b_id = model_b_id
        self.detailed_instructions = detailed_instructions
        self.response_criteria = response_criteria or ["answer accuracy: the ability to provide a correct and complete answer to the question", 
                                                       "Logic: the ability to reason through the question and provide a logical answer", 
                                                       "Empathy: the ability to sense and understand another person’s feelings, perspectives, and underlying needs, then respond in a way that shows you genuinely “get” where they’re coming from", 
                                                       "Explanation ability: the ability to explain the answer in a way that is easy to understand"]
        self.question_criteria = question_criteria or [
                                                        "Does the question describe a realistic, real-world scenario with enough detail to be actionable (i.e., not overly abstract)?",
                                                        "Is the question squarely focused on the target topic/domain rather than being tangential or unrelated?",
                                                        "Is the question too trivial—could it be answered without meaningful expertise?",
                                                        "Is the question intentionally crafted to probe (or exploit) a known weakness or gap in the component’s knowledge?"
                                                        ]

        self.results = {"round_judgments": {}}
        
        # Create the Pydantic parser
        self.parser = PydanticOutputParser(pydantic_object=JudgmentCriteria)
        
        # Create the prompt template for structured output
        self.prompt_template = PromptTemplate(
            template=(
                "You are evaluating a knowledge battle between two AI models. on the topic of {topic} which is {detailed_instructions}\n"
                "Judge the round based on these criteria: {response_criteria}.\n\n"
                "Round transcript:\n{transcript}\n\n"
                "{format_instructions}\n\n"
                "Remember, your evaluation must be thorough but the output format must strictly follow the specified JSON schema."
            ),
            input_variables=["response_criteria", "transcript"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    def judge_round(self, round_num: int) -> Dict[str, Any]:
        """
        Have judges evaluate a specific round in the debate.
        
        Args:
            round_num: The round number to evaluate
            
        Returns:
            Dictionary with judgments for this round
        """
        round_entries = [entry for entry in self.transcript if entry["round"] == round_num]
        
        if not round_entries:
            raise ValueError(f"No entries found for round {round_num}")
        
        round_judgments = {}
        round_votes = {self.model_a_id: 0, self.model_b_id: 0} # votes for the round winner
        round_q_votes = {self.model_a_id: 0, self.model_b_id: 0} # votes for the better question
        
        # Format the round transcript for evaluation
        round_text = self._format_round(round_entries)
        
        for judge in self.judges:
            # Format the prompt with the criteria and transcript
            prompt = self.prompt_template.format(
                topic=self.topic,
                detailed_instructions=self.detailed_instructions,
                response_criteria=", ".join(self.response_criteria),
                transcript=round_text
            )
            
            # Create the context for the judge
            context = {
                "system_prompt": "You are a fair and impartial judge evaluating AI responses. You must provide your evaluation in the exact structured format requested." ,
                "input": prompt
            }
            
            # Get judgment for this round
            judgment_raw = judge.generate_response(context)
            
            try:
                # Use the parser to extract the structured output
                judgment = self.parser.parse(judgment_raw)
                
                # Store the structured judgment
                round_judgments[judge.model_id] = judgment.dict()
                
                
                # Count the vote for the response
                if judgment.winner == "A":
                    round_votes[self.model_a_id] += 1
                elif judgment.winner == "B":
                    round_votes[self.model_b_id] += 1
                
                # Count the vote for the question
                if judgment.better_question == "A":
                    round_q_votes[self.model_a_id] += 1
                elif judgment.better_question == "B":
                    round_q_votes[self.model_b_id] += 1

            except Exception as e:
                # If parsing fails, store the raw judgment and don't count the vote
                round_judgments[judge.model_id] = {
                    "raw_judgment": judgment_raw,
                    "parse_error": str(e)
                }
        
        # Determine round winner for the response
        if round_votes[self.model_a_id] > round_votes[self.model_b_id]:
            round_winner = self.model_a_id
        elif round_votes[self.model_b_id] > round_votes[self.model_a_id]:
            round_winner = self.model_b_id
        else:
            round_winner = "tie"
        
        # Determine round winner for the question
        if round_q_votes[self.model_a_id] > round_q_votes[self.model_b_id]:
            better_question_asker = self.model_a_id
        elif round_q_votes[self.model_b_id] > round_q_votes[self.model_a_id]:
            better_question_asker = self.model_b_id
        else:
            better_question_asker = "tie"
        
        # Store results for this round
        round_results = {
            "judgments": round_judgments,
            "votes": round_votes,
            "winner": round_winner,
            "question_votes": round_q_votes,
            "better_question_asker": better_question_asker
        }
        
        # Update overall results
        self.results["round_judgments"][round_num] = round_results
        
        return round_results
    
    
    def run_battle(self) -> Dict[str, Any]:
        """
        Run the entire battle evaluation, judging each round and providing a final assessment.
        
        Returns:
            Dictionary with complete battle results including round judgments and final assessment
        """
        # Get all unique round numbers from the transcript
        round_numbers = sorted(set(entry["round"] for entry in self.transcript))
        
        # Judge each round
        for round_num in round_numbers:
            self.judge_round(round_num)
        
        # Prepare for final assessment
        battle_summary = {
            "model_a_wins": 0,
            "model_b_wins": 0,
            "ties": 0
        }
        question_summary = {
            "model_a_wins": 0,
            "model_b_wins": 0,
            "ties": 0
        }
        
        # Count round wins
        for round_num, round_result in self.results["round_judgments"].items():
            if round_result["winner"] == self.model_a_id:
                battle_summary["model_a_wins"] += 1
            elif round_result["winner"] == self.model_b_id:
                battle_summary["model_b_wins"] += 1
            else:
                battle_summary["ties"] += 1
            
            if round_result["better_question_asker"] == self.model_a_id:
                question_summary["model_a_wins"] += 1
            elif round_result["better_question_asker"] == self.model_b_id:
                question_summary["model_b_wins"] += 1
            else:
                question_summary["ties"] += 1
        
        
        final_parser = PydanticOutputParser(pydantic_object=FinalAssessment)
        
        # Create prompt template for final assessment
        final_prompt_template = PromptTemplate(
            template=(
                "You are evaluating the entire knowledge battle between two AI models in the topic of {topic} which is {detailed_instructions}.\n"
                "Here is a summary of the round results:\n{round_summary}\n\n"
                "Complete battle transcript:\n{full_transcript}\n\n"
                "Provide a final assessment of which model performed better overall and which model "
                "better utilized the conversation history to create targeted challenging questions.\n\n"
                "{format_instructions}\n\n"
                "Your assessment should be thorough and fair, paying special attention to how models used "
                "information from previous rounds to formulate better questions or attacks."
            ),
            input_variables=["round_summary", "full_transcript", "topic", "detailed_instructions"],
            partial_variables={"format_instructions": final_parser.get_format_instructions()}
        )
        
        # Format the round summary
        round_summary_text = []
        for round_num in round_numbers:
            result = self.results["round_judgments"][round_num]
            winner = "Model A" if result["winner"] == self.model_a_id else "Model B" if result["winner"] == self.model_b_id else "Tie"
            round_summary_text.append(f"Round {round_num}: Winner - {winner}")
        
        # Create the full transcript formatted text
        full_transcript_text = []
        for round_num in round_numbers:
            round_entries = [entry for entry in self.transcript if entry["round"] == round_num]
            round_text = self._format_round(round_entries)
            full_transcript_text.append(f"--- ROUND {round_num} ---\n\n{round_text}")
        
        final_assessments = {}
        for judge in self.judges:
            # Format the prompt with the round summary and transcript
            prompt = final_prompt_template.format(
                round_summary="\n".join(round_summary_text),
                full_transcript="\n\n".join(full_transcript_text),
                topic=self.topic,
                detailed_instructions=self.detailed_instructions
            )
            
            # Create the context for the judge
            context = {
                "system_prompt": "You are a fair and impartial judge evaluating the overall performance in an AI knowledge battle. You must provide your final assessment in the exact structured format requested.",
                "input": prompt
            }
            
            # Get final assessment
            assessment_raw = judge.generate_response(context)
            
            try:
                # Use the parser to extract the structured output
                assessment = final_parser.parse(assessment_raw)
                
                # Store the structured assessment
                final_assessments[judge.model_id] = assessment.dict()
            except Exception as e:
                # If parsing fails, store the raw assessment
                final_assessments[judge.model_id] = {
                    "raw_assessment": assessment_raw,
                    "parse_error": str(e)
                }
        
        # Count final votes
        final_votes = {self.model_a_id: 0, self.model_b_id: 0, "tie": 0}
        history_usage_votes = {self.model_a_id: 0, self.model_b_id: 0}
        
        for judge_id, assessment in final_assessments.items():
            if isinstance(assessment, dict) and "final_winner" in assessment:
                if assessment["final_winner"] == "A":
                    final_votes[self.model_a_id] += 1
                elif assessment["final_winner"] == "B":
                    final_votes[self.model_b_id] += 1
                else:
                    final_votes["tie"] += 1
                    
                if assessment["better_history_user"] == "A":
                    history_usage_votes[self.model_a_id] += 1
                elif assessment["better_history_user"] == "B":
                    history_usage_votes[self.model_b_id] += 1
        
        # Determine overall winner
        if final_votes[self.model_a_id] > final_votes[self.model_b_id] and final_votes[self.model_a_id] > final_votes["tie"]:
            overall_winner = self.model_a_id
        elif final_votes[self.model_b_id] > final_votes[self.model_a_id] and final_votes[self.model_b_id] > final_votes["tie"]:
            overall_winner = self.model_b_id
        else:
            overall_winner = "tie"
        
        # Determine better question asker
        if question_summary["model_a_wins"] > question_summary["model_b_wins"] and question_summary["model_a_wins"] > question_summary["ties"]:
            better_question_asker = self.model_a_id
        elif question_summary["model_b_wins"] > question_summary["model_a_wins"] and question_summary["model_b_wins"] > question_summary["ties"]:
            better_question_asker = self.model_b_id
        else:
            better_question_asker = "tie"
        
        # Determine better history user
        better_history_user = self.model_a_id if history_usage_votes[self.model_a_id] > history_usage_votes[self.model_b_id] else self.model_b_id
        
        # Final results
        final_results = {
            "round_results": self.results["round_judgments"],
            "battle_summary": battle_summary,
            "question_summary": question_summary,
            "final_assessments": final_assessments,
            "final_votes": final_votes,
            "history_usage_votes": history_usage_votes,
            "overall_winner": overall_winner,
            "better_history_user": better_history_user,
            "better_question_asker": better_question_asker
        }
        
        self.results.update(final_results)
        
        return self.results
    
    def _format_round(self, round_entries: List[Dict]) -> str:
        """Format a single round's transcript for evaluation."""
        formatted = []
        
        for entry in round_entries:
            participant_id = entry["participant"]
            role = entry["role"]
            response = entry["response"]
            
            model_label = "Model A" if participant_id == self.model_a_id else "Model B"
            formatted.append(f"{model_label} ({role}): {response}")
        
        return "\n\n".join(formatted)