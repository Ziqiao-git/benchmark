import os
import json
import math

# --------------------------------------------------------
# 1. Set up your models and Elo
# --------------------------------------------------------
MODELS = [
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

# Global Elo ratings. Start everyone at 1200 by default.
elo_ratings = {m: 1200 for m in MODELS}

# Elo update parameters
K_FACTOR = 32

def expected_score(rA, rB):
    """Compute expected score for A vs B."""
    return 1 / (1 + 10 ** ((rB - rA) / 400))

def update_elo(rA, rB, scoreA, scoreB, k=K_FACTOR):
    """Return updated Elo ratings (newA, newB)."""
    eA = expected_score(rA, rB)
    eB = expected_score(rB, rA)
    newA = rA + k * (scoreA - eA)
    newB = rB + k * (scoreB - eB)
    return newA, newB

# Helper to handle a single round's votes
def process_round_votes(round_votes):
    """
    round_votes is a dict like:
      {
        "openrouter-claude-3.7-sonnet-thinking": 5,
        "openrouter-deepseek-v3-0324": 0
      }
    Return (winner, loser) or (None, None) if tie or invalid.
    """
    if len(round_votes) != 2:
        # unexpected number of participants; skip
        return None, None

    # Extract exactly two models
    (modelA, votesA), (modelB, votesB) = list(round_votes.items())

    if votesA > votesB:
        return modelA, modelB
    elif votesB > votesA:
        return modelB, modelA
    else:
        return None, None  # tie

# --------------------------------------------------------
# 2. Process each folder (MT_1 to MT_10)
# --------------------------------------------------------
base_dir = os.getcwd()  # or set to wherever your data is
NUM_FOLDERS = 10

for i in range(1, NUM_FOLDERS + 1):
    folder_name = f"MT_{i}_parallel_debate_results"
    folder_path = os.path.join(base_dir, folder_name)

    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}, skipping...")
        continue

    # Load the state.json to get failed pairs
    failed_pairs_path = os.path.join(folder_path, "state.json")
    failed_pairs = []
    if os.path.isfile(failed_pairs_path):
        with open(failed_pairs_path, "r", encoding="utf-8") as f:
            state_data = json.load(f)
            # e.g.: {"failed_pairs": ["modelA__modelB", ...]}
            failed_pairs = state_data.get("failed_pairs", [])

    # ----------------------------------------------------
    # 3. Iterate over debate_*.json in this folder
    # ----------------------------------------------------
    for fname in os.listdir(folder_path):
        if not fname.endswith(".json"):
            continue
        if not fname.startswith("debate_"):
            continue

        # Extract pair from filename: debate_{A}_vs_{B}.json => "A_vs_B"
        core = fname[len("debate_") : -len(".json")]
        pair_str = core.replace("_vs_", "__")  # e.g. "A_vs_B" => "A__B"

        # Skip if pair is in failed list
        if pair_str in failed_pairs:
            # e.g. "openrouter-Qwen3-235B-A22B__o3"
            continue

        # Load the JSON
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {fname}, JSON parse error: {e}")
            continue

        # If there's an error field, skip
        if "error" in data:
            continue

        # The key we want is data["round_judgments"]
        round_judgments = data.get("round_judgments", {})
        if not round_judgments:
            continue  # no rounds to parse

        # Each round is something like:
        #  "1": {
        #    "votes": { "modelA": X, "modelB": Y },
        #    ...
        #  }
        for round_num_str, round_info in round_judgments.items():
            votes = round_info.get("votes", {})
            winner, loser = process_round_votes(votes)
            if winner is None and loser is None:
                # tie or invalid
                # Elo tie => each gets 0.5
                if len(votes) == 2:
                    (mA, vA), (mB, vB) = list(votes.items())
                    rA = elo_ratings[mA]
                    rB = elo_ratings[mB]
                    newA, newB = update_elo(rA, rB, 0.5, 0.5, K_FACTOR)
                    elo_ratings[mA] = newA
                    elo_ratings[mB] = newB
                continue
            else:
                # winner vs loser
                rW = elo_ratings[winner]
                rL = elo_ratings[loser]
                newW, newL = update_elo(rW, rL, 1.0, 0.0, K_FACTOR)
                elo_ratings[winner] = newW
                elo_ratings[loser] = newL

    # After processing all debates in the folder, save a folder-level result
    sorted_folder_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    folder_result_path = os.path.join(folder_path, "folder_elo_scores.json")
    with open(folder_result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "elo_ratings": {
                    m: round(r, 2) for m, r in elo_ratings.items()
                },
                "ranking": [
                    {"rank": i+1, "model": m, "elo": round(r,2)}
                    for i, (m, r) in enumerate(sorted_folder_elo)
                ]
            },
            f, indent=2
        )
    print(f"[Folder {folder_name}] Elo updated. Wrote {folder_result_path}.")

# --------------------------------------------------------
# 4. After all folders, produce a single overall result
# --------------------------------------------------------
sorted_final_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
overall_path = os.path.join(base_dir, "final_elo_scores.json")
with open(overall_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "elo_ratings": {
                m: round(r, 2) for m, r in elo_ratings.items()
            },
            "ranking": [
                {"rank": i+1, "model": m, "elo": round(r,2)}
                for i, (m, r) in enumerate(sorted_final_elo)
            ]
        },
        f, indent=2
    )
print(f"Final overall Elo written to {overall_path}")