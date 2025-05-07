import os
import json
import random
from typing import List, Dict
from tqdm import tqdm
import time
import asyncio
from async_orchestration import AsyncDebate_and_Judge

from model_interactions import ModelParticipant
from elo import update_elo

def stable_round_robin_elo(
    base_models: List[str],
    judges: List[str],
    project_name: str,
    instruction_set: List[str],
    mass_voter_threshold: float=0.8,
    initial_rounds: int=3,
    max_rounds: int=15,
    k_factor: int=32

) -> Dict[str, float]:
    """
    1) Initialize Elo = 1500
    2) For each pair in round-robin, 
       keep re-trying with +2 rounds if uncertain,
       up to 'max_rounds' 
       then do Elo update if we get 'A','B'
       if still uncertain => treat as tie
    3) Return final Elo dict 
       (You can sort by Elo if you like)
    """
    # Initialize Elo
    if os.path.exists(f"{project_name}/ranking_state.json"):
        with open(f"{project_name}/ranking_state.json", "r") as f:
            state = json.load(f)
        round_robin_state = state["ranking_base"]
        if round_robin_state:
            print("Resuming from previous state:", round_robin_state)
        else:
            print("Starting from scratch of base models ranking")
            round_robin_state = {
                "elo":{m:1500.0 for m in base_models}, #model_id:elo
                "outcomes":{} #pair_key:outcome
            }
    else:
        raise ValueError(f"No ranking_state.json found in {project_name}, need for base models ranking")
    for i in tqdm(range(len(base_models)), desc="Base models ranking"):
        for j in tqdm(range(i+1, len(base_models)), desc=f"RoundRobin Iteration {i+1}"):
            A = base_models[i]
            B = base_models[j]
            sorted_pair = tuple(sorted([A,B]))
            pair_key    = f"{sorted_pair[0]}__vs__{sorted_pair[1]}"

            #If we have already done this pair, skip
            if pair_key in round_robin_state["outcomes"]:
                continue
            # Start with initial
            current_rounds = initial_rounds

            # Attempt multiple times until we get a certain outcome or exceed max_rounds
            outcome = "uncertain"
            while outcome == "uncertain" and current_rounds <= max_rounds:
                outcome = run_match_with_threshold(
                    project_name,
                    A,
                    B,
                    instruction_set,
                    rounds=current_rounds,
                    judges_list=judges,  # or some other judge selection
                    mass_voter_threshold=mass_voter_threshold
                )
                if outcome == "uncertain":
                    current_rounds += 2

            # If still uncertain => treat as tie
            if outcome == "uncertain":
                result_a = 0.5
                final_outcome = "tie"
            elif outcome == "A":
                result_a = 1.0
                final_outcome = "A"
            elif outcome == "B":
                result_a = 0.0
                final_outcome = "B"
            else:
                # "tie"
                result_a = 0.5
                final_outcome = "tie"

            #Elo update
            old_elo_a = round_robin_state["elo"][A]
            old_elo_b = round_robin_state["elo"][B]
            new_elo_a, new_elo_b = update_elo(old_elo_a, old_elo_b, result_a, k=k_factor)
            # add to state
            round_robin_state["elo"][A], round_robin_state["elo"][B] = new_elo_a, new_elo_b
            round_robin_state["outcomes"][pair_key] = final_outcome

            with open(f"{project_name}/ranking_state.json", "w") as f:
                json.dump(round_robin_state, f, indent=2)
    return round_robin_state["elo"]


############################################################
# Checking mass-voter threshold from the evaluation results
############################################################
def parse_evaluation_and_determine_winner(
    eval_path: str, 
    mass_voter_threshold: float = 0.8
) -> str:
    """
    Reads 'evaluation_results_...' JSON. 
    1) Checks if there's a stable final aggregator outcome (≥80%).
    2) Also checks the round-sum aggregator from 'battle_summary'.
    3) If they match, return that winner. Otherwise, return 'uncertain'.
    """
    import os, json
    
    # 1) Check file existence / parse
    if not os.path.exists(eval_path):
       return "uncertain"
    try:
        with open(eval_path, "r") as f:
            data = json.load(f)
        results = data["evaluation"]["results"]
    except:
        return "uncertain"

    # 2) Parse the final mass votes
    final_votes = results.get("final_votes")
    if not final_votes:
        return "uncertain"
    
    total = sum(final_votes.values())  # e.g. {A: x, B: y, tie: z}
    if total == 0:
        return "uncertain"

    a_votes = final_votes.get("modelA", 0)
    b_votes = final_votes.get("modelB", 0)
    tie_votes = final_votes.get("tie", 0)

    # If either A or B or tie is ≥ mass_voter_threshold, we call that the "final aggregator outcome"
    final_outcome = "uncertain"
    if a_votes / total >= mass_voter_threshold:
        final_outcome = "A"
    elif b_votes / total >= mass_voter_threshold:
        final_outcome = "B"
    elif tie_votes / total >= mass_voter_threshold:
        final_outcome = "tie"
    # else remain "uncertain"

    # If final_outcome is still "uncertain," no stable mass vote.
    if final_outcome == "uncertain":
        return "uncertain"

    # 3) Check the round-sum aggregator from "battle_summary"
    battle_summary = results.get("battle_summary")
    if not battle_summary:
        return "uncertain"

    # e.g. "model_a_wins": X, "model_b_wins": Y, "ties": T
    model_a_wins = battle_summary.get("model_a_wins", 0)
    model_b_wins = battle_summary.get("model_b_wins", 0)
    # we don't strictly need the "ties" count if we just see if A>B or B>A or ==

    if model_a_wins > model_b_wins:
        round_outcome = "A"
    elif model_b_wins > model_a_wins:
        round_outcome = "B"
    else:
        round_outcome = "tie"

    # 4) Compare final_outcome vs. round_outcome
    if final_outcome == round_outcome:
        # If they match => we trust this is stable
        return final_outcome
    else:
        # If final aggregator & round-sum aggregator disagree => "uncertain"
        return "uncertain"
############################################################
# Checking mass-voter threshold from the evaluation results
############################################################
def parse_evaluation_and_determine_winner(
    eval_path: str, 
    mass_voter_threshold: float = 0.8
) -> str:
    """
    Reads 'evaluation_results_...' JSON. 
    1) Checks if there's a stable final aggregator outcome (≥80%).
    2) Also checks the round-sum aggregator from 'battle_summary'.
    3) If they match, return that winner. Otherwise, return 'uncertain'.
    """
    import os, json
    
    # 1) Check file existence / parse
    if not os.path.exists(eval_path):
       return "uncertain"
    try:
        with open(eval_path, "r") as f:
            data = json.load(f)
        results = data["evaluation"]["results"]
    except:
        return "uncertain"

    # 2) Parse the final mass votes
    final_votes = results.get("final_votes")
    if not final_votes:
        return "uncertain"
    
    total = sum(final_votes.values())  # e.g. {A: x, B: y, tie: z}
    if total == 0:
        return "uncertain"

    a_votes = final_votes.get("modelA", 0)
    b_votes = final_votes.get("modelB", 0)
    tie_votes = final_votes.get("tie", 0)

    # If either A or B or tie is ≥ mass_voter_threshold, we call that the "final aggregator outcome"
    final_outcome = "uncertain"
    if a_votes / total >= mass_voter_threshold:
        final_outcome = "A"
    elif b_votes / total >= mass_voter_threshold:
        final_outcome = "B"
    elif tie_votes / total >= mass_voter_threshold:
        final_outcome = "tie"
    # else remain "uncertain"

    # If final_outcome is still "uncertain," no stable mass vote.
    if final_outcome == "uncertain":
        return "uncertain"

    # 3) Check the round-sum aggregator from "battle_summary"
    battle_summary = results.get("battle_summary")
    if not battle_summary:
        return "uncertain"

    # e.g. "model_a_wins": X, "model_b_wins": Y, "ties": T
    model_a_wins = battle_summary.get("model_a_wins", 0)
    model_b_wins = battle_summary.get("model_b_wins", 0)
    # we don't strictly need the "ties" count if we just see if A>B or B>A or ==

    if model_a_wins > model_b_wins:
        round_outcome = "A"
    elif model_b_wins > model_a_wins:
        round_outcome = "B"
    else:
        round_outcome = "tie"

    # 4) Compare final_outcome vs. round_outcome
    if final_outcome == round_outcome:
        # If they match => we trust this is stable
        return final_outcome
    else:
        # If final aggregator & round-sum aggregator disagree => "uncertain"
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

    # 1) Check if we have a stable evaluation (For resume check)
    outcome = parse_evaluation_and_determine_winner(evaluation_path, mass_voter_threshold)
    if outcome != "uncertain":
        return outcome  # stable result -> no re-run

    # If not stable, we do the debate asynchronously
    async def _debate_and_judge_async():
        # --------------- Build participants -----------------
        mA = ModelParticipant(model_a_id, role="debater")
        mB = ModelParticipant(model_b_id, role="debater")
        judge_objs = [ModelParticipant(j_id, role="judge") for j_id in judges_list]

        # --------------- Run async debate + judging ---------
        adj = AsyncDebate_and_Judge(
            participants=[mA, mB],
            rounds=rounds,
            instruction_set=instruction_set,
            judges_list=judge_objs,
        )
        results = await adj.run_debate()

        # --------------- Persist results --------------------
        # Save a lightweight transcript file (optional but keeps old path alive)
        with open(debate_results_path, "w", encoding="utf-8") as f:
            json.dump({"transcript": results["transcript"]}, f, indent=2)

        # Save the full evaluation block in the same schema expected by
        # parse_evaluation_and_determine_winner()
        with open(evaluation_path, "w", encoding="utf-8") as f:
            json.dump({"evaluation": {"results": results}}, f, indent=2)

    # Kick off the coroutine (blocks until finished)
    asyncio.run(_debate_and_judge_async())

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
    start_time = time.time()
    # Given a list of all models
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

    topic = "Question that is similar/related to the one in the given instruction "
    detailed_instructions = [
        "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
        "Rewrite your previous response. Start every sentence with the letter A."
    ]
    instruction_set = [topic, detailed_instructions]
    
    # pick Given numer base models
    base_num = 5
    project_name = "MTbench_insert_ranking"
    os.makedirs(project_name, exist_ok=True)
    state_file = os.path.join(project_name, "ranking_state.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            state = json.load(f)
        print("Resuming from previous state:", state)
    else:
        print("No state file found. Starting from scratch.")
        state = {}
    if "shuffle_models" not in state:
        random.shuffle(all_models)
        state["shuffle_models"] = all_models
        state["base_stable_done"] = False
        state["insert_index"] = 0
        with open(state_file, "w") as f:
            json.dump(state, f,indent=2)
    else: 
        all_models = state["shuffle_models"]
    # If base not chosen, choose base_num models
    if "base_models" not in state: 
        state["base_models"] = all_models[:base_num] 
        state["new_models"] = all_models[base_num:]
        state["ranking_base"] = [] # base models has not been ranked yet
        state["ranking_final"] = [] # final ranking has not been ranked yet
        state["insert_index"] = 0 # insert index for new models, now it is 0 since we are not inserting yet
        with open(state_file, "w") as f:
            json.dump(state, f,indent=2)
    base_models = state["base_models"]
    new_models = state["new_models"]
    # If the ranking is not done, we just initialize it as base_models (which is the rank that has not been ranked yet)
    if "ranking_base" not in state:
        state["ranking_base"] = base_models[:]
    if "final_ranking" not in state:
        state["final_ranking"] = base_models[:]
    ranking_base = state["ranking_base"]
    final_ranking = state["final_ranking"]
    insert_index = state["insert_index"]
    base_stable_done = state.get("base_stable_done", False)
        

    # Step1: Stable the base models
    judges = random.sample(all_models, 5)
    if not base_stable_done: # If we haven't stable the base models yet, do it now
        base_model_elo = stable_round_robin_elo(base_models, judges, project_name, instruction_set, initial_rounds=3, max_rounds=9, mass_voter_threshold=0.8)
        sorted_base_model_elo = sorted(base_model_elo.items(), key=lambda x: x[1], reverse=True)
        ranking_base = [model for model, _ in sorted_base_model_elo]
        state["ranking_base"] = ranking_base
        state["base_stable_done"] = True
        with open(state_file, "w") as f:
            json.dump(state, f,indent=2)
        for model in ranking_base:
            print(model, base_model_elo[model])
        print("Base models ranked")

    # Step2: For each new model in new_ones, do insertion, start from insert_index
    final_ranking = ranking_base[:]

    for idx in tqdm(range(insert_index, len(new_models)), desc="Inserting new models"):
        model = new_models[idx]
        judges = state["inserting_judges"][new_models[idx]]
        if judges is None:
            judges = random.sample(all_models, base_num) # choose base_num judges
            state["inserting_judges"][new_models[idx]] = judges
            with open(state_file, "w") as f:
                json.dump(state, f,indent=2)
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

        #update state
        state["ranking_final"] = final_ranking
        state["insert_index"] = idx + 1
        with open(state_file, "w") as f:
            json.dump(state, f,indent=2)

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
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds")

if __name__=="__main__":
    main()
