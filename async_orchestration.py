import asyncio

class AsyncDebate_and_judge:
    def __init__(self, topic, participants, rounds = 3, transcript = [], detailed_instructions = "", evaluation):
        self.topic = topic
        self.participants = participants
        self.evaluation = evaluation  # presumably has async judge methods
        self.rounds = rounds
        self.transcript = transcript
        self.results = {"round_judgments": {}}
        if len(self.participants) != 2:
            raise ValueError("AsyncDebate requires exactly two participants")
    
    async def run_round(self, round_num):
        """
        Run a single round of the debate.
        """
        # Participant A => challenger
        round_entries = []
        
        question_from_a = await self.ask_question(self.participants[0], round_num)
        round_entries.append(question_from_a)

        answer_b = await self.answer_question(self.participants[1], question_from_a, round_num)
        round_entries.append(answer_b)

        question_from_b = await self.ask_question(self.participants[1], round_num)
        round_entries.append(question_from_b)

        answer_a = await self.answer_question(self.participants[0], question_from_b, round_num)
        round_entries.append(answer_a)

        return round_entries

    async def run_debate(self):
        judge_tasks = []
        for round_num in range(1, self.rounds+1):
            # 1) run one round of debate
            entries = await self.run_round(round_num)
            # 2) store
            self.transcript.extend(entries)
            # 3) launch an async judge on just that round’s entries
            judge_tasks.append(asyncio.create_task(self.judge_round_async(round_num, entries)))
        
        # Wait for partial round judgments
        await asyncio.gather(*judge_tasks)
        
        # Then do the final overall judgment
        final_result = await self.judge_final_async()
        self.results["final_assessment"] = final_result
        
        return {
            "transcript": self.transcript,
            "round_judgments": self.results["round_judgments"],
            "final_assessment": final_result
        }

    async def ask_question(self, challenger, round_num):
        # Build context: only previous "challenger" lines from that participant
        prior_questions = self._get_challenger_history_for(challenger, round_num)
        # Format or prompt the model
        question_text = await challenger.async_generate_response({
            "history": prior_questions,
            "system_prompt": "You are the challenger...",
            "input": "Ask a new question..."
        })
        return {
            "round": round_num,
            "role": "challenger",
            "participant": challenger.model_id,
            "response": question_text
        }

    async def answer_question(self, responder, question_entry, round_num):
        # The “input” is the question text from question_entry
        answer_text = await responder.async_generate_response({
            "history": [],  # If you want them to see *none* of their previous answers
            "system_prompt": "You are the responder...",
            "input": f"Please answer this question: {question_entry['response']}"
        })
        return {
            "round": round_num,
            "role": "responder",
            "participant": responder.model_id,
            "response": answer_text
        }

    def _get_challenger_history_for(self, participant, up_to_round):
        # Filter function that returns only this participant's challenger lines < up_to_round
        return [
            e for e in self.transcript
            if e["participant"] == participant.model_id
            and e["role"] == "challenger"
            and e["round"] < up_to_round
        ]
    
    async def judge_round_async(self, round_num, round_entries):
        # call your evaluation object
        judgment = await self.evaluation.judge_round_async(round_num, round_entries)
        self.results["round_judgments"][round_num] = judgment
        
    async def judge_final_async(self):
        # pass the full transcript to get the final verdict
        return await self.evaluation.judge_final_async(self.transcript)