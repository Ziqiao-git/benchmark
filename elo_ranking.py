import os
import json

# --------------------------------------------------------
# 1. Define models and Elo helpers
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

K_FACTOR = 32

def expected_score(rA, rB):
    """Compute expected score of A vs B in Elo."""
    return 1 / (1 + 10 ** ((rB - rA) / 400))

def update_elo(rA, rB, scoreA, scoreB, k=K_FACTOR):
    """Return updated Elo ratings (newA, newB)."""
    eA = expected_score(rA, rB)
    eB = expected_score(rB, rA)
    newA = rA + k * (scoreA - eA)
    newB = rB + k * (scoreB - eB)
    return newA, newB

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
        # unexpected number of participants
        return None, None

    (modelA, votesA), (modelB, votesB) = list(round_votes.items())

    if votesA > votesB:
        return modelA, modelB
    elif votesB > votesA:
        return modelB, modelA
    else:
        return None, None  # tie

def process_debate_file(debate_path, failed_pairs, elo_ratings):
    """
    Reads one debate file, updates `elo_ratings` in-place
    for each round. Skips if in failed_pairs or if error/truncated.
    """
    fname = os.path.basename(debate_path)
    if not fname.startswith("debate_") or not fname.endswith(".json"):
        return

    core = fname[len("debate_"): -len(".json")]
    pair_str = core.replace("_vs_", "__")  # e.g. "o3_vs_o1" => "o3__o1"

    if pair_str in failed_pairs:
        return  # skip

    # Load JSON
    try:
        with open(debate_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Skipping {fname} (JSON parse error): {e}")
        return

    if "error" in data:
        return  # skip if there's an error

    round_judgments = data.get("round_judgments", {})
    if not round_judgments:
        return  # no rounds

    # Process each round
    for round_num_str, round_info in round_judgments.items():
        votes = round_info.get("votes", {})
        winner, loser = process_round_votes(votes)
        if winner is None and loser is None:
            # tie or invalid
            # If tie, each gets 0.5
            if len(votes) == 2:
                (mA, vA), (mB, vB) = list(votes.items())
                rA = elo_ratings[mA]
                rB = elo_ratings[mB]
                newA, newB = update_elo(rA, rB, 0.5, 0.5, K_FACTOR)
                elo_ratings[mA] = newA
                elo_ratings[mB] = newB
        else:
            # winner vs loser
            rW = elo_ratings[winner]
            rL = elo_ratings[loser]
            newW, newL = update_elo(rW, rL, 1.0, 0.0, K_FACTOR)
            elo_ratings[winner] = newW
            elo_ratings[loser] = newL

def write_elo_file(elo_dict, out_path):
    """
    Sort by Elo desc, then write to JSON:
    {
      "elo_ratings": { ... },
      "ranking": [
        {"rank": 1, "model": "some_model", "elo": 1332.52 },
        ...
      ]
    }
    """
    sorted_elo = sorted(elo_dict.items(), key=lambda x: x[1], reverse=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "elo_ratings": {m: round(r, 2) for m, r in elo_dict.items()},
                "ranking": [
                    {"rank": i+1, "model": m, "elo": round(r,2)}
                    for i, (m, r) in enumerate(sorted_elo)
                ]
            },
            f, indent=2
        )
    print(f"Wrote {out_path}")

# --------------------------------------------------------
# Main: Do aggregator + standalone
# --------------------------------------------------------
def main():
    base_dir = os.getcwd()
    NUM_FOLDERS = 10

    # 1) First, do aggregator pass
    aggregator_elo = {m: 1200 for m in MODELS}  # start everyone at 1200
    for i in range(1, NUM_FOLDERS + 1):
        folder_name = f"MT_{i}_9R_9J_parallel_debate_results"
        folder_path = os.path.join(base_dir, folder_name)

        if not os.path.isdir(folder_path):
            print(f"[Aggregator] Folder not found: {folder_path}, skipping...")
            continue

        # Load state.json for failed pairs
        failed_pairs_path = os.path.join(folder_path, "state.json")
        failed_pairs = []
        if os.path.isfile(failed_pairs_path):
            with open(failed_pairs_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)
                failed_pairs = state_data.get("failed_pairs", [])

        # Process each debate JSON in that folder
        for fname in os.listdir(folder_path):
            if fname.endswith(".json") and fname.startswith("debate_"):
                debate_path = os.path.join(folder_path, fname)
                process_debate_file(debate_path, failed_pairs, aggregator_elo)

        # After processing folder i, write aggregator results for folder i
        aggregator_folder_out = os.path.join(folder_path, "folder_elo_scores_aggregated.json")
        write_elo_file(aggregator_elo, aggregator_folder_out)

    # After folder 10, write final aggregator result
    aggregator_final_out = os.path.join(base_dir, "final_elo_scores_aggregated.json")
    write_elo_file(aggregator_elo, aggregator_final_out)
    print("[Aggregator] Done.\n")

    # 2) Next, do standalone pass
    for i in range(1, NUM_FOLDERS + 1):
        folder_name = f"MT_{i}_9R_9J_parallel_debate_results"
        folder_path = os.path.join(base_dir, folder_name)

        if not os.path.isdir(folder_path):
            print(f"[Standalone] Folder not found: {folder_path}, skipping...")
            continue

        # Reset fresh Elo for each folder
        standalone_elo = {m: 1200 for m in MODELS}

        # Load state.json for failed pairs
        failed_pairs_path = os.path.join(folder_path, "state.json")
        failed_pairs = []
        if os.path.isfile(failed_pairs_path):
            with open(failed_pairs_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)
                failed_pairs = state_data.get("failed_pairs", [])

        # Process each debate JSON *just* for that folder
        for fname in os.listdir(folder_path):
            if fname.endswith(".json") and fname.startswith("debate_"):
                debate_path = os.path.join(folder_path, fname)
                process_debate_file(debate_path, failed_pairs, standalone_elo)

        # Write out the standalone result for folder i
        standalone_out = os.path.join(folder_path, "folder_elo_scores_standalone.json")
        write_elo_file(standalone_elo, standalone_out)

    print("[Standalone] Done.")

if __name__ == "__main__":
    main()