import asyncio
from model_interactions import ModelParticipant
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import json
import os
import datetime

class JudgmentCriteria(BaseModel):
    """Standardized judgment format that all judges must follow"""
    winner: Literal["A", "B"] = Field(..., description="Which model performed better (A or B)")
    reason: str = Field(..., description="Detailed explanation for the winner")
    better_question: Literal["A", "B"] = Field(..., description="Which model asked the better question (A or B)")

class AsyncDebate_and_Judge:
    """
    Runs an asynchronous debate between exactly two participants
    using the new 6-step structure: A asks → A answers own → B answers A → B asks → B answers own → A answers B
    and spawns asynchronous round-by-round judgments. 
    After all rounds are done, performs a final/holistic assessment.
    """
    def __init__(
        self,
        participants: List[ModelParticipant],
        rounds: int = 3,
        transcript: Optional[List[Dict]] = None,
        instruction_set: List[str] = None,
        judges_list: List[ModelParticipant] = None,
        response_criteria: Optional[List[str]] = None,
        question_criteria: Optional[List[str]] = None,
        auto_judge: bool = True,
        results_dir: str = "test_results_9"
    ):
        """
        Args:
            participants: Exactly two ModelParticipant instances
            rounds: Number of debate rounds
            transcript: Existing transcript (if any) to extend
            instruction_set: [topic, detailed_instructions]
            judges_list: One or more ModelParticipant used as judges
            response_criteria: Criteria for evaluating the answer portion
            question_criteria: Criteria for evaluating the question portion
            auto_judge: Whether to automatically judge the debate
            results_dir: Directory to save all results (default: "test_results_9")
        """
        if len(participants) != 2:
            raise ValueError("AsyncDebate_and_Judge requires exactly two participants")

        self.participants = participants
        self.model_a_id = self.participants[0].model_id
        self.model_b_id = self.participants[1].model_id

        self.rounds = rounds
        self.transcript = transcript or []
        self.results_dir = results_dir

        # Basic structure for storing results
        self.results = {"round_judgments": {}}

        # instruction_set should have at least [topic, detailed_instructions]
        if not instruction_set or len(instruction_set) < 2:
            instruction_set = ["General Knowledge", "Be creative and thorough."]
        self.topic = instruction_set[0]
        self.detailed_instructions = instruction_set[1]

        # Judges
        self.judges = judges_list or []
        for j in self.judges:
            if not isinstance(j, ModelParticipant):
                raise ValueError("All judges must be ModelParticipant instances")

        # Auto-judge setting
        self.auto_judge = auto_judge

        # Enhanced response criteria that prioritize reasoning over brevity
        self.response_criteria = response_criteria or [
            "REASONING DEPTH: Does the response demonstrate sophisticated analytical thinking, step-by-step logic, and consideration of multiple factors? Higher-quality reasoning should be valued over simple brevity.",
            "TECHNICAL ACCURACY: Are the facts, calculations, numbers, and domain-specific details correct? Precise technical knowledge indicates superior capability.",
            "PROBLEM-SOLVING APPROACH: Does the response break down complex problems systematically? Look for structured thinking, identification of key variables, and logical progression.",
            "DOMAIN EXPERTISE: Does the response demonstrate deep subject-matter knowledge that goes beyond surface-level understanding? Expert-level insights should be rewarded.",
            "PRACTICAL APPLICABILITY: Can the solution actually be implemented? Does it account for real-world constraints and trade-offs?",
            "COMPLETENESS vs CONCISENESS: While responses should be focused, thoroughness in addressing all aspects of a complex question is more valuable than artificial brevity. Judge based on information density, not word count."
        ]

        # Enhanced question criteria
        self.question_criteria = question_criteria or [
            "CONCRETE SPECIFICITY: Does the question include specific numbers, dates, technical terms, standards, or real-world entities?",
            "MULTI-LAYERED COMPLEXITY: Does it require 2-3 steps of reasoning or calculations rather than simple recall?",
            "DOMAIN EXPERTISE TESTING: Does it probe knowledge that only someone with genuine expertise would possess?",
            "PRACTICAL SCENARIO: Does it present a realistic situation someone might actually encounter in the field?",
            "PRECISION REQUIREMENTS: Does it ask for specific values, percentages, procedures, or detailed explanations rather than vague responses?"
        ]

        # Enhanced challenge prompt
        self.challenge_prompt = {
            "system_prompt": (
                f"You are an expert challenger in {self.topic}. Your goal is to craft ONE rigorous, scenario-based question "
                "that will test your opponent's deep understanding and expose potential knowledge gaps.\n\n"
                
                "QUESTION REQUIREMENTS:\n"
                "1. CONCRETE SPECIFICITY: Include specific numbers, dates, technical terms, standards, or real-world entities\n"
                "2. MULTI-LAYERED COMPLEXITY: Require 2-3 steps of reasoning or calculations\n"
                "3. DOMAIN EXPERTISE: Test knowledge that only someone with genuine expertise would know\n"
                "4. PRACTICAL APPLICATION: Present a realistic scenario someone might actually encounter\n"
                "5. PRECISION TESTING: Ask for specific values, percentages, or exact procedures\n\n"
                
                "QUESTION STRUCTURE TEMPLATE:\n"
                "- Context: Set up a specific, realistic scenario with concrete details\n"
                "- Challenge: Present a problem requiring domain knowledge and reasoning\n"
                "- Specificity: Ask for precise calculations, recommendations, or explanations\n"
                "- Verification: Include elements that allow checking the accuracy of the response\n\n"
                
                f"TOPIC-SPECIFIC GUIDANCE FOR {self.topic}:\n"
                f"{self.detailed_instructions}\n\n"
                
                "EXAMPLES OF QUESTION QUALITY:\n\n"
                "❌ POOR (too vague/broad):\n"
                "- 'What can you tell me about battery safety?'\n"
                "- 'How do you handle network security?'\n"
                "- 'What's important in project management?'\n\n"
                
                "✅ EXCELLENT (specific, challenging, testable):\n"
                "- 'A 21700 lithium-ion cell in a battery pack experiences a 15°C rise in core temperature during a 3C discharge at 25°C ambient. "
                "Given IEC 62133 safety standards, calculate the maximum continuous discharge current (in amperes) that would keep the core below 60°C, "
                "and explain which thermal management strategy would be most effective for this configuration.'\n\n"
                
                "- 'Your company's API gateway is receiving 50,000 requests/second with 99.5% hitting cached endpoints. "
                "A DDoS attack increases traffic to 200,000 req/s, with 80% now targeting uncached endpoints. "
                "Calculate the backend load increase factor and recommend specific rate limiting parameters (requests/minute per IP) "
                "to maintain service availability while allowing legitimate traffic.'\n\n"
                
                "STRATEGIC CONSIDERATIONS:\n"
                "- Look for edge cases and boundary conditions in your topic area\n"
                "- Target areas where superficial knowledge fails but deep understanding succeeds\n"
                "- Include industry standards, regulations, or best practices that experts should know\n"
                "- Ask for trade-off analysis between competing approaches\n"
                "- Require application of theoretical knowledge to practical constraints\n\n"
                
                "QUESTION VALIDATION CHECKLIST:\n"
                "□ Does it include specific, verifiable details (numbers, standards, etc.)?\n"
                "□ Would a domain expert need 2-3 minutes of thinking to answer well?\n"
                "□ Can the answer quality be objectively evaluated?\n"
                "□ Does it test practical application, not just memorization?\n"
                "□ Would someone with only surface knowledge struggle?\n\n"
                
                "Now create ONE exceptional question that meets all these criteria and would genuinely challenge an expert in this field."
            ),
            "input": None,
            "history": None,
            "round": None
        }

        # Enhanced response prompt
        self.response_prompt = {
            "system_prompt": (
                f"You are an expert in {self.topic}. Provide a comprehensive, technically accurate response that demonstrates your expertise. "
                "Show your reasoning step-by-step, include specific calculations where needed, and provide practical insights. "
                "Focus on accuracy and depth rather than brevity. Your response should prove your mastery of the subject matter."
            ),
            "input": None,
            "round": None
        }

        # We will need a parser and prompt template for partial round judgments
        self.parser = PydanticOutputParser(pydantic_object=JudgmentCriteria)

        # Enhanced judging prompt template
        self.prompt_template = PromptTemplate(
            template=(
                "You are evaluating a knowledge battle between two AI models on {topic}.\n\n"
                
                "CRITICAL JUDGING INSTRUCTIONS:\n"
                "Your goal is to identify which model demonstrates SUPERIOR REASONING CAPABILITIES, not which gives the shortest answer.\n"
                "Large, advanced models often provide more detailed, nuanced responses that demonstrate deeper understanding.\n"
                "Do not penalize a response for being thorough if that thoroughness adds value.\n\n"
                
                "EVALUATION FRAMEWORK:\n"
                "1. REASONING QUALITY (40% weight):\n"
                "   - Does the model show step-by-step logical thinking?\n"
                "   - Are complex problems broken down systematically?\n"
                "   - Does it consider multiple perspectives or factors?\n"
                "   - Is the analytical approach sophisticated?\n\n"
                
                "2. TECHNICAL PRECISION (30% weight):\n"
                "   - Are calculations and technical details accurate?\n"
                "   - Does it demonstrate genuine domain expertise?\n"
                "   - Are specific standards, protocols, or methodologies correctly referenced?\n\n"
                
                "3. PROBLEM-SOLVING APPROACH (20% weight):\n"
                "   - Is the solution methodology sound?\n"
                "   - Does it address the core challenge effectively?\n"
                "   - Are edge cases and constraints considered?\n\n"
                
                "4. COMMUNICATION EFFECTIVENESS (10% weight):\n"
                "   - Is the explanation clear and well-structured?\n"
                "   - Does it provide actionable insights?\n\n"
                
                "RESPONSE CRITERIA:\n{response_criteria}\n\n"
                
                "ROUND TRANSCRIPT:\n{transcript}\n\n"
                
                "CRITICAL: You must output your evaluation in EXACTLY this JSON format:\n"
                "{format_instructions}\n\n"
                "The winner must be either 'A' or 'B' (not 'Model A' or 'Model B').\n"
                "The better_question must be either 'A' or 'B'.\n"
                "Provide a detailed reason explaining your decision."
            ),
            input_variables=["topic", "response_criteria", "transcript"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    #
    # ---------------------------
    #        File I/O Methods
    # ---------------------------
    #
    def save_debate_transcript(self, filename: str = None):
        """Save debate transcript to JSON"""
        if not filename:
            filename = f"{self.results_dir}/debate_results_{self.model_a_id}_{self.model_b_id}.json"
        
        debate_data = {
            "topic": self.topic,
            "detailed_instructions": self.detailed_instructions,
            "participants": {
                "model_a": self.model_a_id,
                "model_b": self.model_b_id
            },
            "rounds": self.rounds,
            "transcript": self.transcript,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(debate_data, f, indent=2)
        
        print(f"Debate transcript saved to: {filename}")

    def load_debate_transcript(self, filename: str):
        """Load debate transcript from JSON"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.transcript = data["transcript"]
        self.topic = data.get("topic", self.topic)
        self.detailed_instructions = data.get("detailed_instructions", self.detailed_instructions)
        self.rounds = data.get("rounds", self.rounds)
        
        print(f"Debate transcript loaded from: {filename}")

    def save_individual_judgment(self, judge_id: str, round_num: int, judgment_data: Dict, order_suffix: str = "original"):
        """Save individual judge result with order suffix"""
        
        # Create directory structure under main folder
        base_dir = f"{self.results_dir}/judge_results_{self.model_a_id}_{self.model_b_id}"
        sub_dir = f"round_{round_num}_{order_suffix}"
        
        os.makedirs(f"{base_dir}/{sub_dir}", exist_ok=True)
        
        # Create filename
        filename = f"{base_dir}/{sub_dir}/{judge_id}_judgment.json"
        
        # Prepare data
        individual_data = {
            "judge_id": judge_id,
            "round_num": round_num,
            "order": order_suffix,
            "topic": self.topic,
            "participants": {
                "model_a": self.model_a_id,
                "model_b": self.model_b_id
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "judgment": judgment_data
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(individual_data, f, indent=2)
        
        return filename

    def load_individual_judgment(self, judge_id: str, round_num: int, order_suffix: str = "original"):
        """Load individual judge result from JSON file"""
        
        base_dir = f"{self.results_dir}/judge_results_{self.model_a_id}_{self.model_b_id}"
        sub_dir = f"round_{round_num}_{order_suffix}"
        
        filename = f"{base_dir}/{sub_dir}/{judge_id}_judgment.json"
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return None

    def save_judgment_results(self, filename: str = None):
        """Save aggregate judgment results to JSON"""
        if not filename:
            filename = f"{self.results_dir}/judgment_results_{self.model_a_id}_{self.model_b_id}.json"
        
        judgment_data = {
            "topic": self.topic,
            "participants": {
                "model_a": self.model_a_id,
                "model_b": self.model_b_id
            },
            "judges": [judge.model_id for judge in self.judges],
            "round_judgments": self.results["round_judgments"],
            "individual_files_directory": f"{self.results_dir}/judge_results_{self.model_a_id}_{self.model_b_id}",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(judgment_data, f, indent=2)
        
        print(f"Judgment results saved to: {filename}")

    def get_judge_results_summary(self):
        """Get summary of all individual judge results"""
        base_dir = f"{self.results_dir}/judge_results_{self.model_a_id}_{self.model_b_id}"
        
        summary = {
            "total_rounds": self.rounds,
            "judges": [judge.model_id for judge in self.judges],
            "round_results": {}
        }
        
        # Check each round
        for round_num in range(1, self.rounds + 1):
            summary["round_results"][round_num] = {
                "original_order": [],
                "switched_order": []
            }
            
            # Check original order
            original_dir = f"{base_dir}/round_{round_num}_original"
            if os.path.exists(original_dir):
                for judge in self.judges:
                    judge_file = f"{original_dir}/{judge.model_id}_judgment.json"
                    if os.path.exists(judge_file):
                        summary["round_results"][round_num]["original_order"].append(judge.model_id)
            
            # Check switched order
            switched_dir = f"{base_dir}/round_{round_num}_switched"
            if os.path.exists(switched_dir):
                for judge in self.judges:
                    judge_file = f"{switched_dir}/{judge.model_id}_judgment.json"
                    if os.path.exists(judge_file):
                        summary["round_results"][round_num]["switched_order"].append(judge.model_id)
        
        return summary

    #
    # ---------------------------
    #        Debate Logic
    # ---------------------------
    #
    async def run_debate(self) -> Dict[str, Any]:
        """
        Orchestrates all debate rounds asynchronously.
        If auto_judge is True, spawns parallel tasks for each round's partial judgment.
        Otherwise, runs debate only.
        """
        if self.auto_judge:
            # Original behavior: run debate and judge together
            judge_tasks = []
            try: 
                for round_num in range(1, self.rounds + 1):
                    # Run the Q&A for this round (sequentially)
                    round_entries = await self.debate_round(round_num)
                    # Add the new entries to the transcript
                    self.transcript.extend(round_entries)

                    # Launch an async judge task for just-finished round
                    t = asyncio.create_task(self.judge_round_async(round_num, round_entries))
                    judge_tasks.append(t)

                # Wait until all partial round judgments are complete
                await asyncio.gather(*judge_tasks)

                # Save results
                self.save_debate_transcript()
                self.save_judgment_results()

                return {
                    "transcript": self.transcript,
                    "round_judgments": self.results["round_judgments"],
                    "status": "debate_and_judging_completed"
                }
            except Exception as e:
                return {
                    "transcript": self.transcript,
                    "round_judgments": self.results.get("round_judgments", {}),
                    "error": str(e),
                    "status": "debate_and_judging_failed"
                }
        else:
            # New behavior: run debate only
            return await self.run_debate_only()

    async def debate_round(self, round_num: int) -> List[Dict]:
        """
        Run a single round with the new 6-step structure:
          1) Participant A asks a question
          2) Participant A answers their own question
          3) Participant B answers A's question
          4) Participant B asks a question
          5) Participant B answers their own question
          6) Participant A answers B's question
        Returns the new transcript entries for that round.
        """
        round_entries = []

        # 1) A asks a question
        question_a = await self.ask_question(self.participants[0], round_num)
        round_entries.append(question_a)

        # 2) A answers their own question
        answer_a_self = await self.answer_own_question(self.participants[0], question_a, round_num)
        round_entries.append(answer_a_self)

        # 3) B answers A's question
        answer_b = await self.answer_opponent_question(self.participants[1], question_a, round_num)
        round_entries.append(answer_b)

        # 4) B asks a question
        question_b = await self.ask_question(self.participants[1], round_num)
        round_entries.append(question_b)

        # 5) B answers their own question
        answer_b_self = await self.answer_own_question(self.participants[1], question_b, round_num)
        round_entries.append(answer_b_self)

        # 6) A answers B's question
        answer_a = await self.answer_opponent_question(self.participants[0], question_b, round_num)
        round_entries.append(answer_a)

        return round_entries

    async def ask_question(self, challenger: ModelParticipant, round_num: int) -> Dict:
        """
        The challenger sees their own prior questions and opponent's answers to their questions
        (strategic information flow). Prompts them to produce a new question for the opponent.
        """
        context = self.challenge_prompt.copy()
        context["history"] = self._format_history_for(challenger, round_num)
        context["input"] = (
            f"Create a challenging question about {self.topic} that "
            "will be difficult for your opponent to answer correctly. "
            "Use any past context you have (only your own questions and opponent's answers to your questions) to refine it."
        )
        context["round"] = round_num

        question_text = await challenger.generate_response_async(context)  # must be an async model call
        return {
            "round": round_num,
            "step": 1 if challenger.model_id == self.model_a_id else 4,
            "role": "challenger",
            "participant": challenger.model_id,
            "response": question_text
        }

    async def answer_own_question(self, participant: ModelParticipant, question_entry: Dict, round_num: int) -> Dict:
        """
        The participant answers their own question. They see their own questions and opponent's answers to their questions.
        """
        context = self.response_prompt.copy()
        context["history"] = self._format_history_for(participant, round_num)
        q_text = question_entry["response"]
        context["input"] = f"Answer the following question about {self.topic}: {q_text}"
        context["round"] = round_num

        answer_text = await participant.generate_response_async(context)  # must be an async call
        return {
            "round": round_num,
            "step": 2 if participant.model_id == self.model_a_id else 5,
            "role": "responder_self",
            "participant": participant.model_id,
            "response": answer_text
        }

    async def answer_opponent_question(self, responder: ModelParticipant, question_entry: Dict, round_num: int) -> Dict:
        """
        The responder answers the opponent's question. They see their own questions and opponent's answers to their questions.
        The 'input' to the responder is the question text from the challenger.
        """
        context = self.response_prompt.copy()
        context["history"] = self._format_history_for(responder, round_num)
        q_text = question_entry["response"]
        context["input"] = f"Answer the following question about {self.topic}: {q_text}"
        context["round"] = round_num

        answer_text = await responder.generate_response_async(context)  # must be an async call
        return {
            "round": round_num,
            "step": 3 if responder.model_id == self.model_b_id else 6,
            "role": "responder",
            "participant": responder.model_id,
            "response": answer_text
        }

    def _get_history_for(self, participant: ModelParticipant, round_num: int) -> List[Dict]:
        """
        Retrieve the conversation history for this participant, showing:
        * The questions that participant has asked in all rounds up to (and including) 'round_num'
        * The opponent's answers to this participant's questions (to allow strategic adaptation)
        
        This means the participant sees:
        - Their own questions
        - Opponent's answers to their own questions
        - BUT NOT: Their own answers to any questions, or opponent's questions
        """
        filtered_history = []
        for entry in self.transcript:
            if entry["round"] <= round_num:
                # Include participant's own questions only
                if entry["participant"] == participant.model_id and entry["role"] == "challenger":
                    filtered_history.append(entry)
                # Include opponent's answers to this participant's questions
                elif (entry["participant"] != participant.model_id and 
                      entry["role"] == "responder" and
                      self._is_answering_participants_question(entry, participant.model_id)):
                    filtered_history.append(entry)
        return filtered_history
    
    def _is_answering_participants_question(self, answer_entry: Dict, participant_id: str) -> bool:
        """
        Check if this answer is responding to a question asked by the specified participant.
        Using step information to accurately match questions and answers.
        """
        # For step 3: Model B answers Model A's question (step 1)
        # We want to know if this answer is responding to participant_id's question
        if answer_entry["step"] == 3 and participant_id == self.model_a_id:
            return True  # Model A's question is being answered
        # For step 6: Model A answers Model B's question (step 4)
        elif answer_entry["step"] == 6 and participant_id == self.model_b_id:
            return True  # Model B's question is being answered
        return False

    def _format_history_for(self, participant: ModelParticipant, round_num: int) -> List[Dict]:
        """
        Format the history for the model in the expected format.
        """
        history_entries = self._get_history_for(participant, round_num)
        formatted = []
        
        # Check if we have appended any user lines yet
        user_seen = False

        for entry in history_entries:
            same_participant = entry["participant"] == participant.model_id
            
            if same_participant and entry["role"] == "challenger":
                # If no user message has ever been appended, then first ensure
                # the first line after system is user
                if not user_seen:
                    formatted.append({"user": "Please propose a question now."})
                    user_seen = True
                formatted.append({"assistant": entry["response"]})
                
            elif not same_participant and entry["role"] == "responder":
                # This is the opponent's answer to this participant's question
                # Format it as a user message showing the opponent's response
                formatted.append({"user": f"Your previous question was answered as follows: {entry['response']}"})
                user_seen = True
                
            # Note: All responder and responder_self entries are excluded from history

        return formatted

    #
    # ---------------------------
    #         Judging Logic
    # ---------------------------
    #
    async def judge_round_async(self, round_num: int, round_entries: List[Dict]) -> Dict[str, Any]:
        """Judge a single round with both original and switched response orders"""
        
        print(f"Judging round {round_num} with order switching...")
        
        # Get the two responses for this round
        responses = self._extract_responses_from_round(round_entries)
        
        if len(responses) != 2:
            print(f"Warning: Expected 2 responses for round {round_num}, got {len(responses)}")
            return {
                "error": f"Invalid number of responses: {len(responses)}",
                "round_num": round_num
            }
        
        # Judge with original order (A first, B second)
        print(f"  Judging round {round_num} with original order (A first, B second)...")
        original_judgments = await self._judge_responses_with_order(
            round_num, responses, order="original"
        )
        
        # Judge with switched order (B first, A second)
        print(f"  Judging round {round_num} with switched order (B first, A second)...")
        switched_judgments = await self._judge_responses_with_order(
            round_num, responses, order="switched"
        )
        
        # Combine and average the results
        combined_judgments = self._combine_judgment_results(
            original_judgments, switched_judgments, round_num
        )
        
        # Store in results
        self.results["round_judgments"][round_num] = combined_judgments
        
        print(f"Round {round_num} judging completed with order bias analysis")
        
        return combined_judgments

    def _extract_responses_from_round(self, round_entries: List[Dict]) -> Dict[str, str]:
        """Extract the two responses from a round"""
        responses = {}
        
        for entry in round_entries:
            if entry["role"] == "responder" or entry["role"] == "responder_self":
                participant = entry["participant"]
                if participant == self.model_a_id:
                    responses["A"] = entry["response"]
                elif participant == self.model_b_id:
                    responses["B"] = entry["response"]
        
        return responses

    def _create_ordered_transcript(self, responses: Dict[str, str], first: str, second: str) -> str:
        """Create transcript with responses in specified order"""
        
        first_response = responses.get(first, "")
        second_response = responses.get(second, "")
        
        transcript = f"""Model {first} Response:
{first_response}

Model {second} Response:
{second_response}"""
        
        return transcript

    def _detect_order_bias(self, original: Dict, switched: Dict) -> Dict[str, Any]:
        """Detect if there's bias based on response order"""
        
        original_a_wins = original["votes"][self.model_a_id]
        original_b_wins = original["votes"][self.model_b_id]
        switched_a_wins = switched["votes"][self.model_a_id]
        switched_b_wins = switched["votes"][self.model_b_id]
        
        # Check if results are significantly different between orders
        original_diff = abs(original_a_wins - original_b_wins)
        switched_diff = abs(switched_a_wins - switched_b_wins)
        
        bias_detected = abs(original_diff - switched_diff) > 1  # More than 1 vote difference
        
        return {
            "bias_detected": bias_detected,
            "original_order_difference": original_diff,
            "switched_order_difference": switched_diff,
            "consistency_score": 1.0 - (abs(original_diff - switched_diff) / max(original_diff + switched_diff, 1))
        }

    async def _judge_responses_with_order(self, round_num: int, responses: Dict[str, str], order: str) -> Dict[str, Any]:
        """Judge responses with specific order (original or switched)"""
        
        # Create transcript with specified order
        if order == "original":
            # A first, B second
            ordered_transcript = self._create_ordered_transcript(responses, "A", "B")
            order_suffix = "original"
        else:
            # B first, A second
            ordered_transcript = self._create_ordered_transcript(responses, "B", "A")
            order_suffix = "switched"
        
        # Judge with this order
        round_judgments = {}
        round_votes = {self.model_a_id: 0, self.model_b_id: 0}
        round_q_votes = {self.model_a_id: 0, self.model_b_id: 0}
        
        for judge in self.judges:
            # Build the prompt with ordered transcript
            prompt = self.prompt_template.format(
                topic=self.topic,
                response_criteria="\n".join([f"- {criterion}" for criterion in self.response_criteria]),
                transcript=ordered_transcript
            )
            
            judge_context = {
                "system_prompt": (
                    "You are an expert judge evaluating AI model capabilities. Your goal is to identify superior reasoning and expertise, "
                    "not to prefer shorter responses. Advanced models often provide more sophisticated, detailed analysis that should be rewarded. "
                    "Focus on technical accuracy, logical depth, and problem-solving capability. Provide evaluation in exact JSON format."
                ),
                "input": prompt
            }
            
            try:
                judgment_raw = await asyncio.wait_for(
                    judge.generate_response_async(judge_context),
                    timeout=90,
                )
            except asyncio.TimeoutError:
                judgment_data = {"error": "timeout"}
                round_judgments[judge.model_id] = judgment_data
                # Save individual timeout result with order suffix
                self.save_individual_judgment(judge.model_id, round_num, judgment_data, order_suffix)
                continue
            
            try:
                judgment = self.parser.parse(judgment_raw)
                judgment_data = judgment.dict()
                round_judgments[judge.model_id] = judgment_data
                
                # Save individual result with order suffix
                self.save_individual_judgment(judge.model_id, round_num, judgment_data, order_suffix)
                
                # Tally votes (need to map back to original A/B if order was switched)
                if order == "original":
                    winner = judgment.winner
                    better_question = judgment.better_question
                else:
                    # If order was switched, swap A/B back
                    winner = "B" if judgment.winner == "A" else "A"
                    better_question = "B" if judgment.better_question == "A" else "A"
                
                if winner == "A":
                    round_votes[self.model_a_id] += 1
                elif winner == "B":
                    round_votes[self.model_b_id] += 1
                
                if better_question == "A":
                    round_q_votes[self.model_a_id] += 1
                elif better_question == "B":
                    round_q_votes[self.model_b_id] += 1
                    
            except Exception as e:
                judgment_data = {
                    "raw_judgment": judgment_raw,
                    "parse_error": str(e)
                }
                round_judgments[judge.model_id] = judgment_data
                self.save_individual_judgment(judge.model_id, round_num, judgment_data, order_suffix)
        
        return {
            "judgments": round_judgments,
            "votes": round_votes,
            "question_votes": round_q_votes,
            "order": order
        }

    def _combine_judgment_results(self, original: Dict, switched: Dict, round_num: int) -> Dict[str, Any]:
        """Combine results from both orders"""
        
        # Combine votes
        combined_votes = {
            self.model_a_id: original["votes"][self.model_a_id] + switched["votes"][self.model_a_id],
            self.model_b_id: original["votes"][self.model_b_id] + switched["votes"][self.model_b_id]
        }
        
        combined_q_votes = {
            self.model_a_id: original["question_votes"][self.model_a_id] + switched["question_votes"][self.model_a_id],
            self.model_b_id: original["question_votes"][self.model_b_id] + switched["question_votes"][self.model_b_id]
        }
        
        # Determine winners
        if combined_votes[self.model_a_id] > combined_votes[self.model_b_id]:
            round_winner = self.model_a_id
        elif combined_votes[self.model_a_id] < combined_votes[self.model_b_id]:
            round_winner = self.model_b_id
        else:
            round_winner = "tie"
        
        if combined_q_votes[self.model_a_id] > combined_q_votes[self.model_b_id]:
            better_question_asker = self.model_a_id
        elif combined_q_votes[self.model_a_id] < combined_q_votes[self.model_b_id]:
            better_question_asker = self.model_b_id
        else:
            better_question_asker = "tie"
        
        return {
            "judgments": {
                "original_order": original["judgments"],
                "switched_order": switched["judgments"]
            },
            "votes": combined_votes,
            "winner": round_winner,
            "question_votes": combined_q_votes,
            "better_question_asker": better_question_asker,
            "order_analysis": {
                "original_votes": original["votes"],
                "switched_votes": switched["votes"],
                "order_bias_detected": self._detect_order_bias(original, switched)
            }
        }

    async def run_debate_only(self) -> Dict[str, Any]:
        """
        Run all debate rounds sequentially without any judging.
        Save transcript to JSON and return results immediately.
        """
        try:
            for round_num in range(1, self.rounds + 1):
                # Run the Q&A for this round (sequentially)
                round_entries = await self.debate_round(round_num)
                # Add the new entries to the transcript
                self.transcript.extend(round_entries)
                
                print(f"Round {round_num} completed")

            # Save transcript to JSON
            self.save_debate_transcript()
            
            return {
                "transcript": self.transcript,
                "rounds_completed": self.rounds,
                "status": "debate_completed"
            }
        except Exception as e:
            return {
                "transcript": self.transcript,
                "error": str(e),
                "status": "debate_failed"
            }

    async def judge_transcript_only(self, transcript_json_path: str = None) -> Dict[str, Any]:
        """
        Load transcript from JSON and run judging only.
        If no path provided, use current transcript.
        """
        # Load transcript if provided
        if transcript_json_path:
            self.load_debate_transcript(transcript_json_path)
        
        # Spawn async judge tasks for all rounds
        judge_tasks = []
        for round_num in range(1, self.rounds + 1):
            round_entries = [entry for entry in self.transcript if entry["round"] == round_num]
            if round_entries:
                t = asyncio.create_task(self.judge_round_async(round_num, round_entries))
                judge_tasks.append(t)
        
        # Wait for all judgments to complete
        await asyncio.gather(*judge_tasks)
        
        # Save judgment results
        self.save_judgment_results()
        
        return {
            "round_judgments": self.results["round_judgments"],
            "status": "judging_completed"
        }