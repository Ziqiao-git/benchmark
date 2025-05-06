import os
import json
import random
from typing import List, Dict
from tqdm import tqdm

from Debate_orchestration import debate_orchestration
from Judge_orchestration import judge_orchestration
from model_interactions import ModelParticipant

############################################################
# Example ELO helper. You could also do a Bradley-Terry fit.
############################################################
def update_elo(rating_a, rating_b, result_a, k=32):
    """
    rating_a, rating_b: current Elo of model A and model B
    result_a: 1 if A wins, 0 if B wins, 0.5 if tie
    k: step size
    Returns updated rating_a, rating_b
    """
    expected_a = 1.0 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1.0 - expected_a
    new_rating_a = rating_a + k * (result_a - expected_a)
    new_rating_b = rating_b + k * ((1 - result_a) - expected_b)
    return new_rating_a, new_rating_b

############################################################
# Checking mass-voter threshold from the evaluation results
############################################################
def parse_evaluation_and_determine_winner(
    eval_path: str, 
    mass_voter_threshold: float = 0.8
) -> str:
    """
    Reads 'evaluation_results_...' JSON, 
    checks if there's a stable outcome (e.g., 80% votes for A or B).
    Returns 'A','B','tie','uncertain'
    """
    # Load existing evaluation
    if not os.path.exists(eval_path):
        return "uncertain"
    try:
        with open(eval_path, "r") as f:
            data = json.load(f)
        results = data["evaluation"]["results"]
    except:
        return "uncertain"

    # Suppose you store the final votes in something like
    # results["final_votes"] = {modelA_id: X, modelB_id: Y, "tie": Z}
    # This is just an example. You’ll adapt it to however your code does.
    final_votes = results.get("final_votes")
    if not final_votes:
        return "uncertain"
    
    # Example: final_votes = { "modelA": 8, "modelB": 2, "tie": 0 }
    # We see total = 10
    total = sum(final_votes.values())
    if total == 0:
        return "uncertain"

    # Check if any model has >= mass_voter_threshold
    a_votes = final_votes.get("modelA", 0)
    b_votes = final_votes.get("modelB", 0)
    tie_votes = final_votes.get("tie", 0)

    # For example, if A has >=80% => winner = A
    if a_votes / total >= mass_voter_threshold:
        return "A"
    if b_votes / total >= mass_voter_threshold:
        return "B"
    # If tie is big enough or we have < threshold
    # we might call it 'tie' if tie is the biggest chunk
    # or 'uncertain' if no side is above threshold
    # This is just an example policy:
    if tie_votes / total >= mass_voter_threshold:
        return "tie"
    
    # Otherwise, not stable
    return "uncertain"

############################################################
# Debate + Judge with Resume + threshold
############################################################
def run_match_with_threshold(
    project_name: str,
    model_a_id: str,
    model_b_id: str,
    instruction_set: List[str],
    rounds: int = 3,
    judges_list: List[str] = None,
    mass_voter_threshold: float = 0.8,
):
    """
    1) Resume check if evaluation stable
    2) If uncertain, run the debate and judge
    3) parse again
    4) Return final outcome: "A","B","tie","uncertain"
    """
    debates_folder = f"{project_name}/debates"
    judgements_folder = f"{project_name}/judgements"
    os.makedirs(debates_folder, exist_ok=True)
    os.makedirs(judgements_folder, exist_ok=True)

    debate_results_path = f"{debates_folder}/debate_results_{model_a_id}_{model_b_id}.json"
    evaluation_path = f"{judgements_folder}/evaluation_results_{model_a_id}_{model_b_id}.json"

    # 1) Check if we have a stable evaluation
    outcome = parse_evaluation_and_determine_winner(evaluation_path, mass_voter_threshold)
    if outcome != "uncertain":
        return outcome  # stable result -> no re-run

    # If not stable, we do the debate:
    # Also do a "resume check" for debate results
    if not os.path.exists(debate_results_path):
        # create participants
        mA = ModelParticipant(model_a_id, role="debater")
        mB = ModelParticipant(model_b_id, role="debater")
        # single-round debate or multi-round if you want
        debate_orchestration(
            participants=[mA, mB],
            topic= instruction_set[0],
            rounds=rounds,
            project_name=project_name,
            detailed_instructions=instruction_set[1]
        )
    
    # 2) Judge
    judges = []
    for judge in judges_list:
        judges.append(ModelParticipant(judge, role="judge"))
    print("Using these 5 judges:", judges_list)
    judge_orchestration(debate_results_path, judges, project_name)

    # 3) parse again
    outcome2 = parse_evaluation_and_determine_winner(evaluation_path, mass_voter_threshold)
    return outcome2

############################################################
# Placing new model via Binary Search + local bubble
############################################################
def insert_model_with_binary_search(
    model_id: str,
    ranked_list: List[str],
    project_name: str,
    instruction_set,
    rounds: int = 3,
    judges: List[str] = None,
    mass_voter_threshold=0.8,
):
    """
    Insert model_id into ranked_list using:
    - binary search 
    - run_match_with_threshold for each midpoint
    - bubble (local neighbor checks)
    returns the new ranked_list
    """

    if not ranked_list:
        return [model_id]

    left, right = 0, len(ranked_list) - 1
    final_pos = None
    round_num = rounds
    while left <= right:
        mid = (left + right)//2
        mid_model = ranked_list[mid]
        for i in range(5):
            outcome = run_match_with_threshold(
                project_name,
                model_id,
                mid_model,
                instruction_set,
                rounds=round_num,
                judges=judges,
                mass_voter_threshold=mass_voter_threshold
            )
            # interpret outcome => if "A" means model_id better, if "B" means mid_model better
            # depends on how we label them. Let's say run_match sees model_a => model_id, model_b => mid_model
            if outcome == "A":  
                # new model is better -> go left half
                right = mid - 1
                final_pos = mid  # potential position
                break
            elif outcome == "B":
                # new model is worse -> go right half
                left = mid + 1
                final_pos = left
                break
            else:
                round_num += 2
        if final_pos is None:
            final_pos = mid + 1
            left = right + 1

    if final_pos > len(ranked_list):
        final_pos = len(ranked_list)

    # Insert the model in that final_pos
    ranked_list.insert(final_pos, model_id)

    # Local bubble up or down checking neighbors
    # e.g. check above neighbor
    new_index = final_pos
    # bubble up
    while new_index > 0:
        neighbor_index = new_index - 1
        neighbor_model = ranked_list[neighbor_index]
        outcome = run_match_with_threshold(
            project_name,
            model_id,  # "A"
            neighbor_model,  # "B"
            instruction_set,
            mass_voter_threshold
        )
        if outcome == "A":
            # means new_model is better => swap
            ranked_list[neighbor_index], ranked_list[new_index] = ranked_list[new_index], ranked_list[neighbor_index]
            new_index = neighbor_index
        else:
            break

    # might also bubble down if we want
    new_index2 = new_index
    while new_index2 < len(ranked_list)-1:
        neighbor_index = new_index2 + 1
        neighbor_model = ranked_list[neighbor_index]
        outcome = run_match_with_threshold(
            project_name,
            model_id,  # "A"
            neighbor_model, # "B"
            instruction_set,
            mass_voter_threshold
        )
        if outcome == "B":
            # means neighbor is better => swap
            ranked_list[neighbor_index], ranked_list[new_index2] = ranked_list[new_index2], ranked_list[neighbor_index]
            new_index2 = neighbor_index
        else:
            break

    return ranked_list

############################################################
# Example code to do the entire flow:
# Step1) pick 5 random base
# Step2) stabilize them
# Step3) for each new model, insert
# Step4) produce final Elo
############################################################
def main():
    # Suppose we have a list of all models
    all_models = [
        "openrouter-claude-3.7-sonnet-thinking", 
        "openrouter-deepseek-v3-0324", 
        "openrouter-Grok-3-Beta", 
        "openrouter-Gemini-2.5-flash-thinking", 
        "openrouter-QwQ-32B", 
        "openrouter-Qwen3-235B-A22B", 
        "openrouter-Gemini-2.5-pro",
        "o1",
        "o3",
        "o4-mini",
        "deepseek",
        "openrouter-Amazon_Nova_1"
    ]
    random.shuffle(all_models)
    # pick 5 base
    base_5 = all_models[:5]
    new_ones = all_models[5:]
    print("Base 5:", base_5)
    print("Remaining:", new_ones)
    project_name = "MTbench_insert_ranking"
    os.makedirs(project_name, exist_ok=True)
    # Define the instruction set
    topic = "Question that is similar/related to the one in the given instruction "
    detailed_instructions = [
        "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
        "Rewrite your previous response. Start every sentence with the letter A."
    ]
    instruction_set = [topic, detailed_instructions]
    # Step1: rank the base 5 => do repeated round-robin until stable
    # for simplicity, we do 1 pass, but you can do multiple passes
    stable = False
    max_iterations = 5
    iteration = 0
    round_num = 3
    judges = base_5
    while not stable and iteration < max_iterations:
        stable = True
        # do round-robin
        for i in range(len(base_5)):
            for j in range(i+1, len(base_5)):
                outcome = run_match_with_threshold(
                    project_name,
                    base_5[i],
                    base_5[j],
                    instruction_set,
                    rounds=round_num,
                    judges=judges,
                    mass_voter_threshold=0.8
                )
                if outcome == "uncertain":
                    stable = False
        iteration += 1
        round_num += 2
        # possibly reorder them with any rating system
        # for a quick hack, we won't reorder. In a real approach, you'd parse outcomes => update rating => sort
        # if rating stable => stable = True

    # Suppose we store the base_5 in a final "ranking_base"
    ranking_base = base_5[:]  # keep it as is or after your rating-based sorting

    # Step2: For each new model in new_ones, do insertion
    final_ranking = ranking_base[:]

    for model in new_ones:
        judges = random.sample(final_ranking, 5)
        final_ranking = insert_model_with_binary_search(
            model_id=model,
            ranked_list=final_ranking,
            project_name=project_name,
            instruction_set=instruction_set,
            rounds=3,
            judges=judges,
            mass_voter_threshold=0.8
        )
        print("After inserting", model, "ranking is now", final_ranking)

    # Step3: produce final Elo from all stable results
    # we can parse all "evaluation_results_{A}_{B}.json" in project_name/judgements
    # then do a simple Elo aggregator. We'll do a short snippet:

    # Initialize
    elo = {m: 1500 for m in final_ranking}
    # parse each evaluation file
    # for each modelA, modelB outcome, do update_elo(elo[A], elo[B], resultA)
    judgements_path = f"{project_name}/judgements"
    if os.path.exists(judgements_path):
        for fn in os.listdir(judgements_path):
            if fn.startswith("evaluation_results_") and fn.endswith(".json"):
                # parse outcome
                eval_file = os.path.join(judgements_path, fn)
                with open(eval_file, "r") as f:
                    data = json.load(f)
                # Suppose there's a final_winner in data["evaluation"]["results"]["overall_winner"]
                # if final_winner == modelA => result_a=1
                # if modelB => result_a=0
                # if tie => 0.5
                # We'll just do a small example
                results = data["evaluation"]["results"]
                overall_winner = results.get("overall_winner","tie")
                # we need to parse modelA, modelB from the filename maybe
                base_name = fn.replace("evaluation_results_","").replace(".json","")
                parts = base_name.split("_")
                if len(parts)>=2:
                    A_id, B_id = parts[0], parts[1]
                    if A_id in elo and B_id in elo:
                        if overall_winner == A_id:
                            newA, newB = update_elo(elo[A_id], elo[B_id], 1.0)
                            elo[A_id], elo[B_id] = newA, newB
                        elif overall_winner == B_id:
                            newA, newB = update_elo(elo[A_id], elo[B_id], 0.0)
                            elo[A_id], elo[B_id] = newA, newB
                        else:
                            newA, newB = update_elo(elo[A_id], elo[B_id], 0.5)
                            elo[A_id], elo[B_id] = newA, newB

    # final elo-based ranking
    sorted_elo = sorted(elo.items(), key=lambda x: x[1], reverse=True)
    print("\n=== FINAL ELO SCORES ===")
    for m,score in sorted_elo:
        print(m, round(score,2))

if __name__=="__main__":
    main()
