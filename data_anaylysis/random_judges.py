#!/usr/bin/env python3
"""
sample_and_average_win_rates_per_file.py

For each debate_<...>.json file in <results_folder>, do:

1) Parse that file, collecting:
   - Round-level judge votes: "A", "B", or "tie"
   - The set of all judges *in that file*

2) For subset sizes [3,5,7,9,11,13,15] (up to the file's total # of judges):
   - Randomly sample N judges from that file's judge set (10 times)
   - For each subset, re-tally who won each round
   - Compute "A" win fraction among decided rounds (ignore ties)
   - Average across those 10 subsets

3) Print a table of results *just for that file*.

Usage:
  python sample_and_average_win_rates_per_file.py <results_folder>
"""

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

# Configuration
RANDOM_SEED = 42
NUM_SUBSAMPLES = 10    # how many random subsets we try for each size
random.seed(RANDOM_SEED)

def get_subset_winner(judge_votes, chosen_judges):
    """
    judge_votes: dict {judge_id: {"winner": "A"/"B"/"tie"}} for one round
    chosen_judges: subset of judge_ids to consider

    Returns "A", "B", or "tie", or None if no valid votes from chosen_judges.
    """
    tally = Counter()
    for j in chosen_judges:
        if j in judge_votes and "winner" in judge_votes[j]:
            w = judge_votes[j]["winner"]
            tally[w] += 1

    if not tally or sum(tally.values()) == 0:
        return None

    a_count = tally.get("A", 0)
    b_count = tally.get("B", 0)
    tie_count = tally.get("tie", 0)

    # majority vote:
    if a_count > b_count and a_count > tie_count:
        return "A"
    elif b_count > a_count and b_count > tie_count:
        return "B"
    else:
        return "tie"


def analyze_single_file(json_path):
    """
    Parse a single debate_*.json file.
    Identify all judges, round info, then do random subset sampling.
    Print results for this file.
    """
    # 1) Parse JSON
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Skipping {json_path.name} (parse error: {e})")
        return

    if "round_judgments" not in data:
        print(f"No round_judgments in {json_path.name}; skipping.")
        return

    # 2) Collect judges and round-level data
    all_judges = set()
    round_records = []
    for rnd_str, rnd_info in data["round_judgments"].items():
        jdict = rnd_info.get("judgments", {})
        judge_ids = list(jdict.keys())
        for jj in judge_ids:
            all_judges.add(jj)
        round_records.append({
            "round_num": rnd_str,
            "judge_votes": jdict
        })

    all_judges = sorted(all_judges)
    if not round_records:
        print(f"No rounds in {json_path.name}; skipping.")
        return

    # 3) Decide which subset sizes to attempt
    max_j = len(all_judges)
    candidate_sizes = [3, 5, 7, 9, 11, 13, 15]
    subset_sizes = [s for s in candidate_sizes if s <= max_j]
    # Optionally include exact "max_j" if you want:
    # if max_j not in subset_sizes:
    #     subset_sizes.append(max_j)

    # 4) For each subset size, do 10 samples, compute average "A" win rate
    results = []
    for n in subset_sizes:
        a_win_rates = []
        for _ in range(NUM_SUBSAMPLES):
            chosen_judges = random.sample(all_judges, n)
            a_wins = 0
            decided = 0
            # Tally the partial winner each round
            for rr in round_records:
                w = get_subset_winner(rr["judge_votes"], chosen_judges)
                if w == "A":
                    a_wins += 1
                    decided += 1
                elif w == "B":
                    decided += 1
                # tie -> ignore for "decided" fraction

            if decided > 0:
                a_win_rates.append(a_wins / decided)
            else:
                a_win_rates.append(0.0)

        avg_win = statistics.mean(a_win_rates)
        std_win = statistics.pstdev(a_win_rates)
        results.append((n, avg_win, std_win))

    # 5) Print results for this file
    print(f"\n=== Results for file: {json_path.name} ===")
    print(f"Total Judges: {max_j}, Rounds: {len(round_records)}")
    print("Subset Size | Avg A-Win% | StdDev")
    print("--------------------------------")
    for (n, avg, std) in results:
        print(f"{n:^11} | {avg*100:8.2f} | {std*100:7.2f}")


def main(results_folder):
    folder_path = Path(results_folder)
    if not folder_path.is_dir():
        raise SystemExit(f"Error: {results_folder} is not a valid directory.")

    # For each JSON file matching debate_*.json, analyze individually
    files_found = list(folder_path.glob("debate_*.json"))
    if not files_found:
        print(f"No debate_*.json files found in {folder_path}")
        return

    for json_file in sorted(files_found):
        analyze_single_file(json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="For each debate JSON, sample subsets of judges and compute average A win rate."
    )
    parser.add_argument("results_folder", help="Folder containing debate_*.json files")
    args = parser.parse_args()

    main(args.results_folder)