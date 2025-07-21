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

def extract_judge_votes(round_judgments, model_a_id, model_b_id, judge_name):
    """
    Extract specific judge's votes from round judgments.
    Returns dict with model votes based on the specified judge's decisions.
    """
    judge_votes = {model_a_id: 0, model_b_id: 0}
    
    for round_num_str, round_info in round_judgments.items():
        print(f"    Processing round {round_num_str}...")
        
        # Handle session-based format
        if "session_1_judgment" in round_info or "session_2_judgment" in round_info:
            # Session-based format
            for session_key in ["session_1_judgment", "session_2_judgment"]:
                if session_key in round_info:
                    session_data = round_info[session_key]
                    
                    # Check both original and switched orders
                    for order_key in ["original_order", "switched_order"]:
                        if ("judgments" in session_data and 
                            order_key in session_data["judgments"] and
                            judge_name in session_data["judgments"][order_key]):
                            
                            judge_judgment = session_data["judgments"][order_key][judge_name]
                            
                            # Skip if there's a parse error
                            if "parse_error" in judge_judgment:
                                print(f"      Skipping {session_key} {order_key} due to parse error")
                                continue
                            
                            if "winner" in judge_judgment:
                                winner = judge_judgment["winner"]
                                
                                # Map A/B back to actual model names
                                if winner == "A":
                                    judge_votes[model_a_id] += 1
                                elif winner == "B":
                                    judge_votes[model_b_id] += 1
                                
                                print(f"      {session_key} {order_key}: {judge_name} chose {winner}")
        
        # Handle old format (direct votes)
        elif "judgments" in round_info and judge_name in round_info["judgments"]:
            judge_judgment = round_info["judgments"][judge_name]
            
            if "parse_error" in judge_judgment:
                print(f"      Skipping round {round_num_str} due to parse error")
                continue
            
            if "winner" in judge_judgment:
                winner = judge_judgment["winner"]
                
                if winner == "A":
                    judge_votes[model_a_id] += 1
                elif winner == "B":
                    judge_votes[model_b_id] += 1
                
                print(f"      Round {round_num_str}: {judge_name} chose {winner}")
    
    return judge_votes

def process_judge_votes(judge_votes):
    """
    Process judge's votes to determine winner/loser or tie.
    Returns (winner, loser) or (None, None) if tie.
    """
    if len(judge_votes) != 2:
        return None, None

    (modelA, votesA), (modelB, votesB) = list(judge_votes.items())

    if votesA > votesB:
        return modelA, modelB
    elif votesB > votesA:
        return modelB, modelA
    else:
        return None, None  # tie

def process_debate_file_judge_only(debate_path, failed_pairs, elo_ratings, judge_name):
    """
    Process a debate file using only the specified judge's judgments.
    """
    fname = os.path.basename(debate_path)
    if not fname.startswith("debate_") or not fname.endswith(".json"):
        return

    # Extract model names from filename
    core = fname[len("debate_"): -len(".json")]
    if "_vs_" in core:
        model_a_id, model_b_id = core.split("_vs_")
    else:
        print(f"    Could not parse model names from {fname}")
        return

    pair_str = core.replace("_vs_", "__")
    if pair_str in failed_pairs:
        print(f"    Skipping {fname} (in failed pairs)")
        return

    # Load JSON
    try:
        with open(debate_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"    Skipping {fname} (JSON parse error): {e}")
        return

    if "error" in data:
        print(f"    Skipping {fname} (contains error)")
        return

    round_judgments = data.get("round_judgments", {})
    if not round_judgments:
        print(f"    Skipping {fname} (no round judgments)")
        return

    print(f"  Processing {fname} ({judge_name} judgments only)")
    
    # Extract judge's votes
    judge_votes = extract_judge_votes(round_judgments, model_a_id, model_b_id, judge_name)
    
    if not judge_votes or all(v == 0 for v in judge_votes.values()):
        print(f"    No {judge_name} votes found in {fname}")
        return
    
    # Process judge's overall decision
    winner, loser = process_judge_votes(judge_votes)
    
    if winner is None and loser is None:
        # Tie
        if len(judge_votes) == 2:
            (mA, vA), (mB, vB) = list(judge_votes.items())
            rA = elo_ratings[mA]
            rB = elo_ratings[mB]
            newA, newB = update_elo(rA, rB, 0.5, 0.5, K_FACTOR)
            elo_ratings[mA] = newA
            elo_ratings[mB] = newB
            print(f"    {judge_name}'s decision: {mA} vs {mB} -> TIE ({vA}-{vB})")
    else:
        # Winner vs loser
        rW = elo_ratings[winner]
        rL = elo_ratings[loser]
        newW, newL = update_elo(rW, rL, 1.0, 0.0, K_FACTOR)
        elo_ratings[winner] = newW
        elo_ratings[loser] = newL
        winner_votes = judge_votes[winner]
        loser_votes = judge_votes[loser]
        print(f"    {judge_name}'s decision: {winner} beats {loser} ({winner_votes}-{loser_votes})")

def write_elo_file(elo_dict, out_path, judge_name):
    """
    Sort by Elo desc, then write to JSON with judge-specific naming.
    """
    sorted_elo = sorted(elo_dict.items(), key=lambda x: x[1], reverse=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "judge": judge_name,
                "description": f"ELO rankings based solely on {judge_name}'s judgments",
                "elo_ratings": {m: round(r, 2) for m, r in elo_dict.items()},
                "ranking": [
                    {"rank": i+1, "model": m, "elo": round(r,2)}
                    for i, (m, r) in enumerate(sorted_elo)
                ]
            },
            f, indent=2
        )
    print(f"Wrote {judge_name}-only rankings to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Calculate ELO rankings based solely on a specific judge's decisions")
    parser.add_argument("--judge", required=True, help="Name of the judge to use for rankings (e.g., 'deepseek', 'o1', 'gpt4o')")
    parser.add_argument("--standalone", help="Run only standalone pass for specific folders", action="store_true")
    parser.add_argument("--aggregator", help="Run only aggregator pass", action="store_true") 
    parser.add_argument("--folders", nargs="+", help="List of folder paths to process")
    parser.add_argument("--base-dir", help="Base directory (default: current working directory)")
    args = parser.parse_args()

    base_dir = args.base_dir or os.getcwd()
    judge_name = args.judge
    
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

    print("=" * 60)
    print(f"{judge_name.upper()}-ONLY ELO RANKING CALCULATOR")
    print("=" * 60)
    print(f"This script calculates ELO rankings based solely on {judge_name}'s judgments.")
    print("Other judges' decisions are ignored.")
    print()

    # Initialize aggregator Elo if needed
    if run_aggregator:
        print("[AGGREGATOR MODE] - Accumulating ELO across all folders")
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
                debate_path = os.path.join(folder_path, fname)
                process_debate_file_judge_only(debate_path, failed_pairs, aggregator_elo, judge_name)

            # After processing folder, write aggregator results for folder
            aggregator_folder_out = os.path.join(folder_path, f"folder_elo_scores_{judge_name}_only_aggregated.json")
            write_elo_file(aggregator_elo, aggregator_folder_out, judge_name)

        # After all folders, write final aggregator result
        aggregator_final_out = os.path.join(base_dir, f"final_elo_scores_{judge_name}_only_aggregated.json")
        write_elo_file(aggregator_elo, aggregator_final_out, judge_name)
        print("[Aggregator] Done.\n")

    # Standalone pass
    if run_standalone:
        print("[STANDALONE MODE] - Fresh ELO for each folder")
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
                debate_path = os.path.join(folder_path, fname)
                process_debate_file_judge_only(debate_path, failed_pairs, standalone_elo, judge_name)

            # Write out the standalone result for folder
            standalone_out = os.path.join(folder_path, f"folder_elo_scores_{judge_name}_only_standalone.json")
            write_elo_file(standalone_elo, standalone_out, judge_name)

        print("[Standalone] Done.")

if __name__ == "__main__":
    main() 