import os
import json

from Debate_orchestration import debate_orchestration
from Judge_orchestration import judge_orchestration
from model_interactions import ModelParticipant

# 1) A function to update ELO
def update_elo(rating_a, rating_b, result_a, k=32):
    """
    rating_a, rating_b: current Elo ratings for A, B
    result_a: 1.0 if A wins, 0 if B wins, 0.5 if tie
    """
    expected_a = 1 / (1 + 10**((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a
    
    new_rating_a = rating_a + k * (result_a - expected_a)
    new_rating_b = rating_b + k * ((1 - result_a) - expected_b)
    return new_rating_a, new_rating_b


# 2) Participants with default Elo
participants = {
    "o3":      {"elo": 1500},
    "o4-mini": {"elo": 1500},
    "gpt4o":   {"elo": 1500},
    "o1":      {"elo": 1500}
}

participant_ids = list(participants.keys())

# 3) Topic and instructions
topic = "Real Life doctor"
detailed_instructions = "The questioner will take on the role of a patient, seeking advice. The rival will act as a doctor, offering guidance to the patient. The patient poses questions or concerns, and the doctor responds with supportive advice. The primary objective is to evaluate how effectively the participant can provide comfort and reassurance to the patient."

def is_valid_debate_file(filepath):
    """Check if a debate JSON file exists and has at least some expected keys."""
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        # Minimal check: does it have "results" -> "transcript"?
        if "results" in data and "transcript" in data["results"]:
            return True
    except:
        pass
    return False

def is_valid_evaluation_file(filepath):
    """Check if an evaluation JSON file exists and has the 'battle_summary'."""
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        # Minimal check: does it have "evaluation" -> "results" -> "battle_summary"?
        if "evaluation" in data and "results" in data["evaluation"]:
            if "battle_summary" in data["evaluation"]["results"]:
                return True
    except:
        pass
    return False


# 4) Round-robin pairing
for i in range(len(participant_ids)):
    for j in range(i + 1, len(participant_ids)):
        model_a_id = participant_ids[i]
        model_b_id = participant_ids[j]

        debate_results_path = f"{topic}/debates/debate_results_{model_a_id}_{model_b_id}.json"
        evaluation_results_path = f"{topic}/judgements/evaluation_results_{model_a_id}_{model_b_id}.json"

        print(f"\n=== Checking pair {model_a_id} vs {model_b_id} ===")

        # ============== DEBATE STEP ==============
        if is_valid_debate_file(debate_results_path):
            print(f"SKIP debate for {model_a_id} vs {model_b_id} (already done)")
        else:
            print(f"RUN debate for {model_a_id} vs {model_b_id}")
            # Create participants
            model_a = ModelParticipant(model_a_id, role="debater")
            model_b = ModelParticipant(model_b_id, role="debater")
            # Run the debate
            debate_orchestration(
                [model_a, model_b],
                topic,
                rounds=7,
                project_name=topic,
                detailed_instructions=detailed_instructions
            )

        # ============== EVALUATION STEP ==============
        if is_valid_evaluation_file(evaluation_results_path):
            print(f"SKIP evaluation for {model_a_id} vs {model_b_id} (already done)")
        else:
            print(f"RUN evaluation for {model_a_id} vs {model_b_id}")
            judges = [
                ModelParticipant("o1", role="judge"),
                ModelParticipant("gpt4o", role="judge"),
                ModelParticipant("o3", role="judge"),
                ModelParticipant("o4-mini", role="judge"),
            ]
            judge_orchestration(debate_results_path, judges, topic)

        # ============== ELO UPDATE ==============
        # We do Elo update once we know there's a valid evaluation file
        if not is_valid_evaluation_file(evaluation_results_path):
            # If we STILL don't have a valid evaluation, skip Elo update
            print(f"WARNING: No valid evaluation found, skipping Elo for {model_a_id} vs {model_b_id}")
            continue
        # Otherwise, parse it to find the results
        with open(evaluation_results_path, "r") as f:
            data = json.load(f)

        battle_summary = data["evaluation"]["results"].get("battle_summary", {})
        model_a_wins = battle_summary.get("model_a_wins", 0)
        model_b_wins = battle_summary.get("model_b_wins", 0)

        if model_a_wins > model_b_wins:
            result_a = 1.0
        elif model_b_wins > model_a_wins:
            result_a = 0.0
        else:
            result_a = 0.5

        current_elo_a = participants[model_a_id]["elo"]
        current_elo_b = participants[model_b_id]["elo"]
        new_elo_a, new_elo_b = update_elo(current_elo_a, current_elo_b, result_a)
        participants[model_a_id]["elo"] = round(new_elo_a, 2)
        participants[model_b_id]["elo"] = round(new_elo_b, 2)

        print(f"ELO updated -> {model_a_id}: {participants[model_a_id]['elo']}  {model_b_id}: {participants[model_b_id]['elo']}")

# 5) Final Elo Standings
print("\n=== Final Elo Standings ===")
for pid, info in sorted(participants.items(), key=lambda x: x[1]['elo'], reverse=True):
    print(f"{pid}: {info['elo']}")
