import os
import json
import argparse

# --------------------------------------------------------
# 1. Define models and Elo helpers
# --------------------------------------------------------
MODELS = [
    # 1. Deepseek-R1
    "deepseek",
    # 2. O1
    "o1",
    # 3. Qwen3-235B
    "openrouter-Qwen3-235B-A22B",
    # 4. Claude-3.7
    "openrouter-claude-3.7-sonnet-thinking",
    # 5. GPT-4o
    "gpt4o",
    # 6. DeepSeek-V3
    "openrouter-deepseek-v3-0324",
    # 7. Qwen2.5-72B-Instruct
    "openrouter-qwen-2-72b-instruct",
    # 8. llama-3.3-70b-instruct
    "openrouter-meta-llama-llama-3.3-70b-instruct",
    # 9. Claude-3.5
    "openrouter-claude-3.5-haiku",
    # 10. mistralai/mixtral-8x7b-instruct
    "openrouter-mistral-8x7b-instruct",
    # 11. Gemma-2-27B
    "openrouter-google-gemma-2-27b-it",
    # 12. qwen/qwen-2-72b-instruct
    "openrouter-qwen-2-72b-instruct",
    # 13. Mistral-7b-instructv02
    "openrouter-mistralai-mistral-7b-instruct-v0.2",
    # 14. Gemma-2-9B
    "openrouter-google-gemma-2-9b-it",
    # 15. microsoft/phi-4-reasoning-plus
    "openrouter-phi-4-reasoning-plus",
    #16. QwQ-32B
    "openrouter-QwQ-32B",
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
    for each round. Handles both old format and new session-based format.
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

    # Process each round - handle both old and new session-based format
    for round_num_str, round_info in round_judgments.items():
        # Check if this is the new session-based format
        if "round_summary" in round_info and "total_votes" in round_info["round_summary"]:
            # New format: use aggregated votes from round_summary
            votes = round_info["round_summary"]["total_votes"]
        elif "votes" in round_info:
            # Old format: use votes directly
            votes = round_info["votes"]
        else:
            # Try to aggregate session votes if round_summary is missing
            votes = {}
            for session_key, session_data in round_info.items():
                if session_key.endswith("_judgment") and "votes" in session_data:
                    session_votes = session_data["votes"]
                    for model, count in session_votes.items():
                        votes[model] = votes.get(model, 0) + count
        
        if not votes:
            continue  # skip if no votes found
            
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
                print(f"  Round {round_num_str}: {mA} vs {mB} -> TIE ({vA}-{vB})")
        else:
            # winner vs loser
            rW = elo_ratings[winner]
            rL = elo_ratings[loser]
            newW, newL = update_elo(rW, rL, 1.0, 0.0, K_FACTOR)
            elo_ratings[winner] = newW
            elo_ratings[loser] = newL
            winner_votes = votes[winner]
            loser_votes = votes[loser]
            print(f"  Round {round_num_str}: {winner} beats {loser} ({winner_votes}-{loser_votes})")

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
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="Calculate Elo rankings for debate results")
    parser.add_argument("--standalone", help="Run only standalone pass for a specific folder", action="store_true")
    parser.add_argument("--aggregator", help="Run only aggregator pass", action="store_true") 
    parser.add_argument("--folders", nargs="+", help="List of folder paths to process (default: Math_3R_5J_1 to Math_3R_5J_10)")
    parser.add_argument("--base-dir", help="Base directory (default: current working directory)")
    args = parser.parse_args()

    base_dir = args.base_dir or os.getcwd()
    
    # Determine which folders to process
    if args.folders:
        folders = args.folders
    else:
        # Default: look for test_results_* folders
        folders = [d for d in os.listdir(base_dir) if d.startswith("test_results_") and os.path.isdir(os.path.join(base_dir, d))]
        if not folders:
            print("No test_results_* folders found. Please specify --folders explicitly.")
            return
    
    # Determine which passes to run
    if args.standalone or args.aggregator:
        run_aggregator = args.aggregator
        run_standalone = args.standalone
    else:
        # Default: run both if no specific pass is requested
        run_aggregator = True
        run_standalone = True

    # Initialize aggregator Elo if needed
    if run_aggregator:
        aggregator_elo = {m: 1200 for m in MODELS}  # start everyone at 1200
        
        for folder_name in folders:
            folder_path = os.path.join(base_dir, folder_name)

            if not os.path.isdir(folder_path):
                print(f"[Aggregator] Folder not found: {folder_path}, skipping...")
                continue

            print(f"[Aggregator] Processing folder: {folder_name}")

            # Load state.json for failed pairs
            failed_pairs_path = os.path.join(folder_path, "state.json")
            failed_pairs = []
            if os.path.isfile(failed_pairs_path):
                try:
                    with open(failed_pairs_path, "r", encoding="utf-8") as f:
                        state_data = json.load(f)
                        failed_pairs = state_data.get("failed_pairs", [])
                except Exception as e:
                    print(f"  Warning: Could not load state.json: {e}")

            # Process each debate JSON in that folder
            debate_files = [f for f in os.listdir(folder_path) if f.endswith(".json") and f.startswith("debate_")]
            for fname in debate_files:
                print(f"  Processing: {fname}")
                debate_path = os.path.join(folder_path, fname)
                process_debate_file(debate_path, failed_pairs, aggregator_elo)

            # After processing folder, write aggregator results for folder
            aggregator_folder_out = os.path.join(folder_path, "folder_elo_scores_aggregated.json")
            write_elo_file(aggregator_elo, aggregator_folder_out)

        # After all folders, write final aggregator result
        aggregator_final_out = os.path.join(base_dir, "final_elo_scores_aggregated.json")
        write_elo_file(aggregator_elo, aggregator_final_out)
        print("[Aggregator] Done.\n")

    # Standalone pass
    if run_standalone:
        for folder_name in folders:
            folder_path = os.path.join(base_dir, folder_name)

            if not os.path.isdir(folder_path):
                print(f"[Standalone] Folder not found: {folder_path}, skipping...")
                continue

            print(f"[Standalone] Processing folder: {folder_name}")

            # Reset fresh Elo for each folder
            standalone_elo = {m: 1200 for m in MODELS}

            # Load state.json for failed pairs
            failed_pairs_path = os.path.join(folder_path, "state.json")
            failed_pairs = []
            if os.path.isfile(failed_pairs_path):
                try:
                    with open(failed_pairs_path, "r", encoding="utf-8") as f:
                        state_data = json.load(f)
                        failed_pairs = state_data.get("failed_pairs", [])
                except Exception as e:
                    print(f"  Warning: Could not load state.json: {e}")

            # Process each debate JSON *just* for that folder
            debate_files = [f for f in os.listdir(folder_path) if f.endswith(".json") and f.startswith("debate_")]
            for fname in debate_files:
                print(f"  Processing: {fname}")
                debate_path = os.path.join(folder_path, fname)
                process_debate_file(debate_path, failed_pairs, standalone_elo)

            # Write out the standalone result for folder
            standalone_out = os.path.join(folder_path, "folder_elo_scores_standalone.json")
            write_elo_file(standalone_elo, standalone_out)

        print("[Standalone] Done.")

if __name__ == "__main__":
    main()