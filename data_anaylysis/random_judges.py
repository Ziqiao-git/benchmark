#!/usr/bin/env python3
"""
sample_and_average_win_rates.py

Reads all debate_*.json files in <results_folder>. 
Each debate has round-level judge votes for "A"/"B"/"tie".

For each subset size in [3,5,7,9,..., up to total # of judges]:
  1) Perform 10 random samples (subsets) of N judges.
  2) For each subset, re-tally the round winners.
  3) Compute how often the 'A' model wins (ignore ties).
  4) Average that "A wins" fraction over the 10 subsets.

Print a final table: [N, #rounds, average A-win rate, stdev]

Usage:
  python sample_and_average_win_rates.py <results_folder>
"""

import argparse
import json
import math
import os
import random
import statistics
from collections import Counter
from pathlib import Path

# ------------- Configuration -----------------
RANDOM_SEED = 42
NUM_SUBSAMPLES = 10  # how many subsets to sample at each size
random.seed(RANDOM_SEED)
# ---------------------------------------------

def get_subset_winner(judge_votes, chosen_judges):
    """
    judge_votes: dict {judge_id: {"winner": "A"/"B"/"tie"}} for a single round
    chosen_judges: subset of judge_ids to consider

    Returns: "A", "B", or "tie", or None if no votes
    """
    votes = Counter()
    for j in chosen_judges:
        if j in judge_votes and "winner" in judge_votes[j]:
            w = judge_votes[j]["winner"]
            votes[w] += 1

    if not votes or sum(votes.values()) == 0:
        return None

    a_count = votes.get("A", 0)
    b_count = votes.get("B", 0)
    tie_count = votes.get("tie", 0)

    # majority
    if a_count > b_count and a_count > tie_count:
        return "A"
    elif b_count > a_count and b_count > tie_count:
        return "B"
    else:
        return "tie"


def main(folder):
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise SystemExit(f"Error: {folder} is not a directory.")

    # Step 1: Parse all debate JSONs, collect:
    #  - The set of all judge IDs
    #  - Round-level data { full_judge_list, judge_votes, etc. }
    all_judges = set()
    round_records = []

    for file in folder_path.iterdir():
        if not (file.is_file() and file.name.startswith("debate_") and file.name.endswith(".json")):
            continue

        try:
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {file.name} (parse error: {e})")
            continue

        if "round_judgments" not in data:
            continue

        for rnd_str, rnd_info in data["round_judgments"].items():
            jdict = rnd_info.get("judgments", {})
            these_judges = list(jdict.keys())
            for jj in these_judges:
                all_judges.add(jj)

            round_records.append({
                "debate_file": file.name,
                "round_num": rnd_str,
                "judge_votes": jdict,
            })

    all_judges = sorted(all_judges)
    if not round_records:
        print(f"No valid round data found in {folder}. Exiting.")
        return

    print(f"Total rounds found: {len(round_records)}")
    print(f"Total unique judges: {len(all_judges)}")
    print(f"Judge IDs: {all_judges}\n")

    # We'll create a list of subset sizes we want to try, e.g. 3, 5, 7, 9...
    # up to the total number of judges (but skipping if we can't form that subset).
    subset_sizes = []
    step_candidates = [3, 5, 7, 9, 11, 13, 15]
    # If you want to keep going until you reach len(all_judges), do something like:
    # step_candidates = list(range(3, len(all_judges)+1, 2))  # every odd # from 3 up
    # Or mix them. For now, let's see how many judges we have:
    max_judges = len(all_judges)
    for s in step_candidates:
        if s <= max_judges:
            subset_sizes.append(s)
    # If you want to ensure you go all the way to max_judges:
    # if max_judges not in subset_sizes:
    #     subset_sizes.append(max_judges)

    results = []

    # Step 2: For each subset size, do 10 random samples, compute average "A" win rate
    for n in subset_sizes:
        # We will store the fraction of rounds that "A" won for each of the 10 subsets
        a_win_rates = []

        for _ in range(NUM_SUBSAMPLES):
            chosen_judges = random.sample(all_judges, n)
            # Tally how many times "A" is the winner (ignoring ties)
            a_wins = 0
            decided_rounds = 0  # counting only rounds with a non-tie outcome from the subset

            for rr in round_records:
                sub_w = get_subset_winner(rr["judge_votes"], chosen_judges)
                if sub_w == "A":
                    a_wins += 1
                    decided_rounds += 1
                elif sub_w == "B":
                    decided_rounds += 1
                # tie => skip

            # "win rate" = fraction of decided rounds that went to A
            if decided_rounds > 0:
                win_rate = a_wins / decided_rounds
            else:
                win_rate = 0.0

            a_win_rates.append(win_rate)

        # After 10 runs, average them
        avg_win_rate = statistics.mean(a_win_rates)
        stdev_win_rate = statistics.pstdev(a_win_rates)  # or statistics.stdev

        # We'll keep track of the *average decided rounds* across the 10 subsets as well
        # (just to see how many times we get a tie or no votes).
        # But let's do that in a second pass if needed. For simplicity, let's not store it here.

        results.append({
            "n": n,
            "avg_win_rate": avg_win_rate,
            "stdev_win_rate": stdev_win_rate,
        })

    # Step 3: Print results
    print("Subset Size | Avg Win Rate (A) | Std Dev")
    print("---------------------------------------")
    for row in results:
        print(f"{row['n']:^11} | {row['avg_win_rate']*100:7.2f}%       | {row['stdev_win_rate']*100:7.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample subsets of judges and compute average A win rate."
    )
    parser.add_argument("results_folder", help="Folder with debate_*.json files")
    args = parser.parse_args()

    main(args.results_folder)