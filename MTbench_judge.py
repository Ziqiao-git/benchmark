import os
import json
import random
from tqdm import tqdm
from model_interactions import ModelParticipant
from Judge_orchestration import judge_orchestration

#############################################
# ELO HELPER
#############################################
def update_elo(rating_a, rating_b, result_a, k=32):
    """
    rating_a, rating_b: current Elo for A, B
    result_a: 1 if A wins, 0 if B wins, 0.5 if tie
    Returns (new_rating_a, new_rating_b)
    """
    expected_a = 1 / (1 + 10**((rating_b - rating_a)/400))
    expected_b = 1 - expected_a
    new_rating_a = rating_a + k * (result_a - expected_a)
    new_rating_b = rating_b + k * ((1 - result_a) - expected_b)
    return new_rating_a, new_rating_b

def parse_participants_from_filename(base_name):
    """
    Given 'debate_results_{A}_{B}.json', return (A, B).
    If it doesn't start with 'debate_results_', return None.
    """
    if not base_name.startswith("debate_results_") or not base_name.endswith(".json"):
        return None
    # debate_results_{A}_{B}.json
    core = base_name[len("debate_results_"):-5]  # remove prefix & '.json'
    # => '{A}_{B}'
    parts = core.split("_")
    if len(parts) < 2:
        return None
    # everything except last => modelA, last => modelB? or we assume exactly 2?
    # typically it's 2. If underscores in model IDs, you might need more logic.
    # We'll do a simpler approach:
    model_a_id = parts[0]
    model_b_id = "_".join(parts[1:])  # if there's underscores in the second model
    return (model_a_id, model_b_id)


#############################################
# MAIN
#############################################

def main():
    ###########################################
    # 1) CONFIG
    ###########################################
    project_name = "MTbench_1"    # Where debate .json are
    debates_dir = os.path.join(project_name, "debates")
    judgements_dir = os.path.join(project_name, "judgements")
    os.makedirs(judgements_dir, exist_ok=True)

    # FULL set of participants. We read from the debate files or define them.
    # We'll build Elo for all models that appear in the debate folder,
    # or you can define a known set if you prefer.
    # If your code knows the exact participants, you can define them:
    # all_models = [...]
    # We'll do an empty set and fill it from filenames:
    all_models = set()

    # 2) SCAN DEBATE FILES
    debate_files = []
    for fn in os.listdir(debates_dir):
        if fn.startswith("debate_results_") and fn.endswith(".json"):
            debate_files.append(fn)
            # parse participants
            parts = parse_participants_from_filename(fn)
            if parts:
                a_id, b_id = parts
                all_models.add(a_id)
                all_models.add(b_id)
    debate_files.sort()

    all_models = list(all_models)
    print(f"Found {len(debate_files)} debate files. Models discovered: {all_models}")

    # 3) Initialize Elo
    answer_elo = {m: 1500.0 for m in all_models}
    question_elo = {m: 1500.0 for m in all_models}

    # 5) We'll store a set of processed pairs so we don't double-update Elo
    # even if we parse the same file again
    pairs_done = set()

    # 6) Progress bar
    progress = tqdm(total=len(debate_files), desc="Judging & Elo")

    for base_name in debate_files:
        debate_path = os.path.join(debates_dir, base_name)
        # parse participants
        p = parse_participants_from_filename(base_name)
        if not p:
            print(f"Skipping invalid debate file name: {base_name}")
            progress.update(1)
            continue

        model_a_id, model_b_id = p

        # We also check if there's a reversed version we should use for naming
        # but typically if the file is 'debate_results_X_Y.json', we'll do
        # 'evaluation_results_X_Y.json' for the eval.

        # Build the evaluation path
        eval_name = base_name.replace("debate_results_", "evaluation_results_")
        evaluation_path = os.path.join(judgements_dir, eval_name)

        # 7) Resume check for evaluation
        # If it exists and is valid => parse Elo if not done
        done_key = tuple(sorted([model_a_id, model_b_id]))
        if os.path.exists(evaluation_path):
            try:
                with open(evaluation_path, "r") as f:
                    existing_eval = json.load(f)
                if "evaluation" in existing_eval and "results" in existing_eval["evaluation"]:
                    # valid evaluation
                    if done_key not in pairs_done:
                        # parse it to do Elo updates
                        parse_and_update_elo(
                            existing_eval, model_a_id, model_b_id, 
                            answer_elo, question_elo
                        )
                        pairs_done.add(done_key)
                    # skip the judge step
                    progress.update(1)
                    continue
            except:
                pass
            # if corrupted, we'll re-judge

        # =============== DO THE JUDGE ===============
        print(f"Judging debate: {model_a_id} vs {model_b_id}")
        # If you want exactly 5 out of these 6, random or fixed:
        judge_list = random.sample(all_models, 5)
        judges = []
        for judge in judge_list:
            judges.append(ModelParticipant(judge, role="judge"))
        print("Using these 5 judges:", judge_list)
        judge_orchestration(
            json_file_path=debate_path,
            judges=judges,
            project_name=project_name
        )

        # Now parse the new evaluation
        if os.path.exists(evaluation_path):
            # parse & update Elo
            if done_key not in pairs_done:
                with open(evaluation_path, "r") as f:
                    data = json.load(f)
                parse_and_update_elo(data, model_a_id, model_b_id, answer_elo, question_elo)
                pairs_done.add(done_key)

        progress.update(1)

    progress.close()

    # 8) Print final Elo
    print("\n=== FINAL ANSWER ELO ===")
    sorted_ans = sorted(answer_elo.items(), key=lambda x: x[1], reverse=True)
    for mid, rating in sorted_ans:
        print(f"{mid}: {round(rating,2)}")

    print("\n=== FINAL QUESTION ELO ===")
    sorted_q = sorted(question_elo.items(), key=lambda x: x[1], reverse=True)
    for mid, rating in sorted_q:
        print(f"{mid}: {round(rating,2)}")


def parse_and_update_elo(data, model_a_id, model_b_id, answer_elo, question_elo):
    """
    Given loaded evaluation data, parse the final
    'battle_summary' and 'question_summary' from
    evaluation->results. Update answer_elo & question_elo accordingly.
    """
    results = data.get("evaluation", {}).get("results", {})
    battle_summary = results.get("battle_summary", {})
    question_summary = results.get("question_summary", {})

    a_wins_ans = battle_summary.get("model_a_wins", 0)
    b_wins_ans = battle_summary.get("model_b_wins", 0)
    if a_wins_ans > b_wins_ans:
        # A is answer winner
        answer_elo[model_a_id], answer_elo[model_b_id] = update_elo(
            answer_elo[model_a_id],
            answer_elo[model_b_id],
            1.0
        )
    elif b_wins_ans > a_wins_ans:
        answer_elo[model_a_id], answer_elo[model_b_id] = update_elo(
            answer_elo[model_a_id],
            answer_elo[model_b_id],
            0.0
        )
    else:
        # tie
        answer_elo[model_a_id], answer_elo[model_b_id] = update_elo(
            answer_elo[model_a_id],
            answer_elo[model_b_id],
            0.5
        )

    a_wins_q = question_summary.get("model_a_wins", 0)
    b_wins_q = question_summary.get("model_b_wins", 0)
    if a_wins_q > b_wins_q:
        # A better question
        question_elo[model_a_id], question_elo[model_b_id] = update_elo(
            question_elo[model_a_id],
            question_elo[model_b_id],
            1.0
        )
    elif b_wins_q > a_wins_q:
        question_elo[model_a_id], question_elo[model_b_id] = update_elo(
            question_elo[model_a_id],
            question_elo[model_b_id],
            0.0
        )
    else:
        question_elo[model_a_id], question_elo[model_b_id] = update_elo(
            question_elo[model_a_id],
            question_elo[model_b_id],
            0.5
        )


if __name__ == "__main__":
    main()
