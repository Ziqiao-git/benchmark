from Debate_orchestration import debate_orchestration
from Judge_orchestration import judge_orchestration
from model_interactions import ModelParticipant

# 1) A function to update ELO
def update_elo(rating_a, rating_b, result_a, k=32):
    """
    rating_a, rating_b: current Elo ratings for A, B
    result_a: 1.0 if A wins, 0 if B wins, 0.5 if tie
    """
    # Expected scores
    expected_a = 1 / (1 + 10**((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a
    
    new_rating_a = rating_a + k * (result_a - expected_a)
    new_rating_b = rating_b + k * ((1 - result_a) - expected_b)
    return new_rating_a, new_rating_b

# 2) Define your participants with default Elo
participants = {
    "o3":      {"elo": 1500},
    "o4-mini": {"elo": 1500},
    "gpt4o":   {"elo": 1500},
    "o1":      {"elo": 1500}
    # Add as many as you like
}

# Convert to a list if you want an ordered iteration
participant_ids = list(participants.keys())

# 3) Round-robin pairing
topic = "Consoling a friend"
detailed_instructions = (
    "The Questioner will take on the role of a Visitor, seeking advice. "
    "The Rival will act as a Therapist, offering guidance to the Visitor. "
    "The Visitor poses questions or concerns, and the Therapist responds with supportive advice. "
    "The primary objective is to evaluate how effectively the participant can provide comfort."
)

for i in range(len(participant_ids)):
    for j in range(i + 1, len(participant_ids)):
        model_a_id = participant_ids[i]
        model_b_id = participant_ids[j]

        # Create ModelParticipant objects for each debate
        model_a = ModelParticipant(model_a_id, role="debater")
        model_b = ModelParticipant(model_b_id, role="debater")

        print(f"\n=== Starting debate between {model_a_id} and {model_b_id} ===")

        # Run the debate
        debate_orchestration(
            [model_a, model_b],
            topic,
            rounds=3,  # or however many rounds you like
            project_name=topic,
            detailed_instructions=detailed_instructions
        )
        
        # 4) Run the judging
        judges = [
            ModelParticipant("o1", role="judge"),
            ModelParticipant("gpt4o", role="judge"),
            ModelParticipant("o3", role="judge"),
            ModelParticipant("o4-mini", role="judge"),
        ]

        # The JSON path your debate_orchestration created
        debate_results_path = f"{topic}/debates/debate_results_{model_a.model_id}_{model_b.model_id}.json"
        judge_orchestration(debate_results_path, judges, topic)

        # 5) Extract final winner from the evaluation JSON
        #    by reading the file that judge_orchestration wrote
        eval_path = f"{topic}/judgements/evaluation_results_{model_a.model_id}_{model_b.model_id}.json"
        import json
        with open(eval_path, "r") as f:
            data = json.load(f)

        # 1) Extract the battle summary
        battle_summary = data["evaluation"]["results"].get("battle_summary", {})
        model_a_wins = battle_summary.get("model_a_wins", 0)
        model_b_wins = battle_summary.get("model_b_wins", 0)

        # 2) Determine the result based on who has more round wins
        if model_a_wins > model_b_wins:
            # Model A wins overall by round count
            result_a = 1.0
        elif model_b_wins > model_a_wins:
            # Model B wins overall
            result_a = 0.0
        else:
            # If tied in round count
            result_a = 0.5


        # 6) Update Elo ratings
        current_elo_a = participants[model_a_id]["elo"]
        current_elo_b = participants[model_b_id]["elo"]
        new_elo_a, new_elo_b = update_elo(current_elo_a, current_elo_b, result_a)
        participants[model_a_id]["elo"] = round(new_elo_a, 2)
        participants[model_b_id]["elo"] = round(new_elo_b, 2)

        print(f"{model_a_id} new Elo: {participants[model_a_id]['elo']}")
        print(f"{model_b_id} new Elo: {participants[model_b_id]['elo']}")

# 7) Final Elo Standings
print("\n=== Final Elo Standings ===")
for pid, info in sorted(participants.items(), key=lambda x: x[1]['elo'], reverse=True):
    print(f"{pid}: {info['elo']}")