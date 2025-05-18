import asyncio
from model_interactions import ModelParticipant
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

class JudgmentCriteria(BaseModel):
    """
    Evaluation criteria for a round.
    Must produce 'winner' (A/B) for the better answer
    and 'better_question' (A/B) for the better question.
    """
    winner: Literal["A", "B"] = Field(..., description="Which model performed better in answering the rival's question under response criteria (A or B)")
    better_question: Literal["A", "B"] = Field(..., description="Which model asked the better question under question criteria (A or B)")

class FinalAssessment(BaseModel):
    """
    Final assessment of the entire debate.
    Must produce 'final_winner' (A/B) for the overall winner
    and 'better_history_user' (A/B) for who best used historical context to attack.
    """
    better_history_user: Literal["A", "B"] = Field(..., description="Which model better used history to attack (A or B)")
    final_winner: Literal["A", "B"] = Field(..., description="Overall debate winner (A or B)")

class AsyncDebate_and_Judge:
    """
    Runs an asynchronous debate between exactly two participants
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
        question_criteria: Optional[List[str]] = None
    ):
        """
        Args:
            topic: The debate topic
            participants: Exactly two ModelParticipant instances
            rounds: Number of debate rounds
            transcript: Existing transcript (if any) to extend
            instruction_set: [topic, detailed_instructions]
            judges_list: One or more ModelParticipant used as judges
            response_criteria: Criteria for evaluating the answer portion
            question_criteria: Criteria for evaluating the question portion
        """
        if len(participants) != 2:
            raise ValueError("AsyncDebate_and_Judge requires exactly two participants")

        self.participants = participants
        self.model_a_id = self.participants[0].model_id
        self.model_b_id = self.participants[1].model_id

        self.rounds = rounds
        self.transcript = transcript or []

        # Basic structure for storing results
        self.results = {"round_judgments": {}}

        # instruction_set should have at least [topic, detailed_instructions]
        if not instruction_set or len(instruction_set) < 2:
            instruction_set = [topic, "Be creative and thorough."]
        self.topic = instruction_set[0]
        self.detailed_instructions = instruction_set[1]

        # Judges
        self.judges = judges_list or []
        for j in self.judges:
            if not isinstance(j, ModelParticipant):
                raise ValueError("All judges must be ModelParticipant instances")

        # Default criteria
        self.response_criteria = response_criteria or [
            "Answer accuracy: correct and complete",
            "Logic: uses sound reasoning",
            "Empathy: demonstrates understanding of the question context",
            "Explanation: clarity and detail"
        ]
        self.question_criteria = question_criteria or [
            "Realistic, on-topic scenario?",
            "Not overly trivial or tangential?",
            "Probes a known gap or requires domain knowledge?"
        ]

        # Prompts for challengers and responders
        self.challenge_prompt = {
            "system_prompt": (
                f"You are a challenger on the topic of {self.topic}. "
                f"Ask a difficult question that will test your opponent's knowledge. "
                f"Here are some detailed instructions: {self.detailed_instructions}"
            ),
            "input": None,
            "history": None,
            "round": None
        }
        self.response_prompt = {
            "system_prompt": (
                f"You are answering a challenging question on the topic of {self.topic}."
            ),
            "input": None,
            "round": None
        }

        # We will need a parser and prompt template for partial round judgments
        self.parser = PydanticOutputParser(pydantic_object=JudgmentCriteria)

        self.prompt_template = PromptTemplate(
            template=(
                "You are a fair and impartial judge evaluating a knowledge battle between two AI models "
                "on the topic of {topic}, which is {detailed_instructions}.\n\n"
                "Judge this single round based on the following response criteria: {response_criteria}.\n\n"
                "Here is the relevant round transcript:\n{transcript}\n\n"
                "{format_instructions}\n"
                "Remember, your evaluation must follow the JSON schema strictly."
            ),
            input_variables=["topic", "detailed_instructions", "response_criteria", "transcript"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    #
    # ---------------------------
    #        Debate Logic
    # ---------------------------
    #
    async def run_debate(self) -> Dict[str, Any]:
        """
        Orchestrates all debate rounds asynchronously.
        Spawns parallel tasks for each round's partial judgment.
        Concludes with a final holistic evaluation.
        """
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

            # Perform final/holistic assessment once all rounds are done (we do not do this anymore)
            # final_result = await self.judge_final_async()
            # self.results["final_assessment"] = final_result

            return {
                "transcript": self.transcript,
                "round_judgments": self.results["round_judgments"],
                # "final_assessment": final_result
            }
        except Exception as e:
            return {
            "transcript": self.transcript,
            "round_judgments": self.results.get("round_judgments", {}),
            "error": str(e),
        }

    async def debate_round(self, round_num: int) -> List[Dict]:
        """
        Run a single round (two challenges, two answers):
          1) Participant A => challenger, B => answerer
          2) Participant B => challenger, A => answerer
        Returns the new transcript entries for that round.
        """
        round_entries = []

        # 1) A asks, B answers
        question_from_a = await self.ask_question(self.participants[0], round_num)
        round_entries.append(question_from_a)

        answer_b = await self.answer_question(self.participants[1], question_from_a, round_num)
        round_entries.append(answer_b)

        # 2) B asks, A answers
        question_from_b = await self.ask_question(self.participants[1], round_num)
        round_entries.append(question_from_b)

        answer_a = await self.answer_question(self.participants[0], question_from_b, round_num)
        round_entries.append(answer_a)

        return round_entries

    async def ask_question(self, challenger: ModelParticipant, round_num: int) -> Dict:
        """
        The challenger sees ONLY their own prior 'challenger' lines up to this round
        (no answers, no rival's questions).
        Prompts them to produce a new question for the opponent.
        """
        context = self.challenge_prompt.copy()
        context["history"] = self._get_challenger_history_for(challenger, round_num)
        context["input"] = (
            f"Create a challenging question about {self.topic} that "
            "will be difficult for your opponent to answer correctly. "
            "Use any of your own prior questions to refine it."
        )
        context["round"] = round_num

        question_text = await challenger.generate_response_async(context)  # must be an async model call
        return {
            "round": round_num,
            "role": "challenger",
            "participant": challenger.model_id,
            "response": question_text
        }

    async def answer_question(self, responder: ModelParticipant, question_entry: Dict, round_num: int) -> Dict:
        """
        The responder sees only *their own* prior messages (if you want them to see them),
        or you can hide them; current code hides everything except the direct question.
        The 'input' to the responder is the question text from the challenger.
        """
        context = self.response_prompt.copy()
        q_text = question_entry["response"]
        context["input"] = f"Answer the following question about {self.topic}: {q_text}"
        context["round"] = round_num

        answer_text = await responder.generate_response_async(context)  # must be an async call
        return {
            "round": round_num,
            "role": "responder",
            "participant": responder.model_id,
            "response": answer_text
        }

    def _get_challenger_history_for(self, participant: ModelParticipant, up_to_round: int) -> List[Dict]:
        """
        Return all lines where this participant was 'challenger' in previous rounds.
        i.e., A does not see B’s lines, nor does it see its own answers from prior rounds.
        """
        return [
            e for e in self.transcript
            if e["participant"] == participant.model_id
            and e["role"] == "challenger"
            and e["round"] < up_to_round
        ]

    #
    # ---------------------------
    #         Judging Logic
    # ---------------------------
    #
    async def judge_round_async(self, round_num: int, round_entries: List[Dict]) -> Dict[str, Any]:
        """
        Asynchronously calls each judge on the *just finished* round’s transcript.
        Collates the votes for 'winner' and 'better_question' into self.results.
        """
        round_judgments = {}
        # Tally votes (e.g. if you have multiple judges)
        round_votes = {self.model_a_id: 0, self.model_b_id: 0}
        round_q_votes = {self.model_a_id: 0, self.model_b_id: 0}

        # Convert round entries to text for judging
        round_text = self._format_round(round_entries)

        # For each judge, get structured output
        for judge in self.judges:
            # Build the prompt
            prompt = self.prompt_template.format(
                topic=self.topic,
                detailed_instructions=self.detailed_instructions,
                response_criteria=", ".join(self.response_criteria),
                transcript=round_text
            )

            judge_context = {
                "system_prompt": (
                    "You are a fair and impartial judge. Provide your evaluation in the exact JSON format."
                ),
                "input": prompt
            }

            # Call the judge model with a timeout (skip if it hangs)
            try:
                judgment_raw = await asyncio.wait_for(
                    judge.generate_response_async(judge_context),
                    timeout=90,   # seconds; tweak as needed
                )
            except asyncio.TimeoutError:
                round_judgments[judge.model_id] = {"error": "timeout"}
                continue

            try:
                # Parse structured output
                judgment = self.parser.parse(judgment_raw)
                round_judgments[judge.model_id] = judgment.dict()

                # Tally votes
                if judgment.winner == "A":
                    round_votes[self.model_a_id] += 1
                else:
                    round_votes[self.model_b_id] += 1

                if judgment.better_question == "A":
                    round_q_votes[self.model_a_id] += 1
                else:
                    round_q_votes[self.model_b_id] += 1

            except Exception as e:
                # If parsing fails, store raw
                round_judgments[judge.model_id] = {
                    "raw_judgment": judgment_raw,
                    "parse_error": str(e)
                }

        # Decide final round winners
        if round_votes[self.model_a_id] > round_votes[self.model_b_id]:
            round_winner = self.model_a_id
        elif round_votes[self.model_a_id] < round_votes[self.model_b_id]:
            round_winner = self.model_b_id
        else:
            round_winner = "tie"

        if round_q_votes[self.model_a_id] > round_q_votes[self.model_b_id]:
            better_question_asker = self.model_a_id
        elif round_q_votes[self.model_a_id] < round_q_votes[self.model_b_id]:
            better_question_asker = self.model_b_id
        else:
            better_question_asker = "tie"

        round_results = {
            "judgments": round_judgments,
            "votes": round_votes,
            "winner": round_winner,
            "question_votes": round_q_votes,
            "better_question_asker": better_question_asker
        }

        # Store in results
        self.results["round_judgments"][round_num] = round_results
        return round_results

    async def judge_final_async(self) -> Dict[str, Any]:
        """
        Once all rounds are complete, each judge does a holistic final assessment
        of the entire debate (all rounds).
        """
        # Summaries for the “battle” and question usage
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

        # Count how many times each model won in each round
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

        # Prepare final prompt
        final_parser = PydanticOutputParser(pydantic_object=FinalAssessment)
        final_prompt_template = PromptTemplate(
            template=(
                "You are evaluating the entire knowledge battle between two AI models "
                "on the topic of {topic}, which is {detailed_instructions}.\n\n"
                "Here is a summary of round results:\n{round_summary}\n\n"
                "Complete battle transcript:\n{full_transcript}\n\n"
                "Provide a final assessment: which model performed better overall, and which model "
                "better utilized the conversation history to create targeted challenging questions?\n\n"
                "{format_instructions}\n\n"
                "Be thorough and fair. Pay attention to usage of context from previous rounds."
            ),
            input_variables=["topic", "detailed_instructions", "round_summary", "full_transcript"],
            partial_variables={"format_instructions": final_parser.get_format_instructions()}
        )

        # Prepare text for each round
        round_numbers = sorted(self.results["round_judgments"].keys())
        # Summaries
        round_summary_list = []
        for rn in round_numbers:
            r = self.results["round_judgments"][rn]
            if r["winner"] == self.model_a_id:
                w = "Model A"
            elif r["winner"] == self.model_b_id:
                w = "Model B"
            else:
                w = "Tie"
            round_summary_list.append(f"Round {rn}: Winner = {w}")

        # Full transcript text
        full_transcript_list = []
        for rn in round_numbers:
            rn_entries = [e for e in self.transcript if e["round"] == rn]
            round_str = self._format_round(rn_entries)
            full_transcript_list.append(f"--- Round {rn} ---\n{round_str}")

        # Build final prompt text
        round_summary_text = "\n".join(round_summary_list)
        full_transcript_text = "\n\n".join(full_transcript_list)

        # We will gather final_assessments from all judges
        final_assessments = {}

        # Tally votes across judges
        final_votes = {self.model_a_id: 0, self.model_b_id: 0, "tie": 0}
        history_usage_votes = {self.model_a_id: 0, self.model_b_id: 0}

        # Let each judge produce a final verdict
        for judge in self.judges:
            prompt = final_prompt_template.format(
                topic=self.topic,
                detailed_instructions=self.detailed_instructions,
                round_summary=round_summary_text,
                full_transcript=full_transcript_text
            )

            judge_context = {
                "system_prompt": (
                    "You are a fair and impartial judge evaluating overall performance. "
                    "Output must follow the specified JSON schema exactly."
                ),
                "input": prompt
            }

            try:
                assessment_raw = await asyncio.wait_for(
                    judge.generate_response_async(judge_context),
                    timeout=120,  # holistic judgment might take longer
                )
            except asyncio.TimeoutError:
                final_assessments[judge.model_id] = {"error": "timeout"}
                continue
            try:
                final_struct = final_parser.parse(assessment_raw)
                final_assessments[judge.model_id] = final_struct.dict()

                # Tally the final_winner
                if final_struct.final_winner == "A":
                    final_votes[self.model_a_id] += 1
                elif final_struct.final_winner == "B":
                    final_votes[self.model_b_id] += 1
                else:
                    final_votes["tie"] += 1

                # Tally the better_history_user
                if final_struct.better_history_user == "A":
                    history_usage_votes[self.model_a_id] += 1
                else:
                    history_usage_votes[self.model_b_id] += 1

            except Exception as e:
                final_assessments[judge.model_id] = {
                    "raw_assessment": assessment_raw,
                    "parse_error": str(e)
                }

        # Compute overall final_winner
        if final_votes[self.model_a_id] > final_votes[self.model_b_id] and final_votes[self.model_a_id] > final_votes["tie"]:
            overall_winner = self.model_a_id
        elif final_votes[self.model_b_id] > final_votes[self.model_a_id] and final_votes[self.model_b_id] > final_votes["tie"]:
            overall_winner = self.model_b_id
        else:
            overall_winner = "tie"

        # Compute better_history_user (just whichever got the most votes)
        # If tie, choose one or store "tie" as well—feel free to handle it differently.
        if history_usage_votes[self.model_a_id] > history_usage_votes[self.model_b_id]:
            better_history_user = self.model_a_id
        elif history_usage_votes[self.model_b_id] > history_usage_votes[self.model_a_id]:
            better_history_user = self.model_b_id
        else:
            better_history_user = "tie"

        final_results = {
            "round_results": self.results["round_judgments"],  # detailed round info
            "battle_summary": battle_summary,
            "question_summary": question_summary,
            "final_assessments": final_assessments,     # raw final outputs from judges
            "final_votes": final_votes,
            "history_usage_votes": history_usage_votes,
            "overall_winner": overall_winner,
            "better_history_user": better_history_user
        }

        # Update self.results
        self.results.update(final_results)
        return final_results

    def _format_round(self, round_entries: List[Dict]) -> str:
        """
        Turn a single round's Q/A exchanges into a readable text block for the judge.
        (Similar to the synchronous version in model_interactions.)
        """
        lines = []
        for e in round_entries:
            pid = e["participant"]
            role = e["role"]
            if pid == self.model_a_id:
                model_label = "Model A"
            elif pid == self.model_b_id:
                model_label = "Model B"
            else:
                model_label = pid  # fallback
            lines.append(f"{model_label} ({role}): {e['response']}")
        return "\n".join(lines)