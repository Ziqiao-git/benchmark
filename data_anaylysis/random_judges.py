#!/usr/bin/env python3

"""
sample_and_average_win_rates_per_folder_final_winner_plot.py

Generates ONE aggregated plot per folder, summarizing how
final-winner fractions vary with subset size across all debate files.

Steps:
1) For each debate_*.json in <results_folder>:
   - Determine the unique final winner (model that wins the most rounds),
     or skip the file if there's no single winner.
   - Collect the number of judges and do subset sampling for N in [3,5,7,...]
     (up to that file's judge count).
   - For each N, do 10 random subsets, compute fraction of decided rounds
     that prefer the final winner => "file-level" fraction for that subset size.
2) Aggregate these fractions across all valid files:
   - For each subset size, we gather the fractions from each file
     and compute a MEAN and STDDEV across files.
3) Produce ONE plot (folder_plot.png) in <results_folder>:
   - x-axis = subset size
   - y-axis = average fraction (with error bars = std dev)
4) Print aggregated results in the terminal.

Usage:
  python sample_and_average_win_rates_per_folder_final_winner_plot.py <results_folder>

Requires:
  pip install matplotlib
"""

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # so it can run without an interactive display
import matplotlib.pyplot as plt

RANDOM_SEED = 42
NUM_SUBSAMPLES = 10
random.seed(RANDOM_SEED)

def get_round_winner_all_judges(judge_votes):
    """
    Return the majority winner across all judges in one round:
      - model label (e.g. "A", "B", "Claude", etc.)
      - "tie" if there's no strictly greater count
      - None if no votes
    """
    tally = Counter()
    for j, info in judge_votes.items():
        w = info.get("winner")
        if w is not None:
            tally[w] += 1

    if not tally:
        return None

    best_label, best_count = None, 0
    multiple_best = False
    for label, count in tally.items():
        if count > best_count:
            best_label = label
            best_count = count
            multiple_best = False
        elif count == best_count:
            multiple_best = True

    if multiple_best or best_label == "tie":
        return "tie"
    return best_label

def get_subset_round_winner(judge_votes, chosen_judges):
    """
    Return the majority winner across chosen_judges in one round.
    Same logic as get_round_winner_all_judges, but restricted subset.
    """
    tally = Counter()
    for j in chosen_judges:
        if j in judge_votes:
            w = judge_votes[j].get("winner")
            if w is not None:
                tally[w] += 1

    if not tally:
        return None

    best_label, best_count = None, 0
    multiple_best = False
    for label, count in tally.items():
        if count > best_count:
            best_label = label
            best_count = count
            multiple_best = False
        elif count == best_count:
            multiple_best = True

    if multiple_best or best_label == "tie":
        return "tie"
    return best_label


def analyze_single_file(json_path):
    """
    Return a dict mapping subset_size -> average fraction for that file,
    or None if the file has no unique final winner.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Skipping {json_path.name} (parse error: {e})")
        return None

    if "round_judgments" not in data:
        print(f"No round_judgments in {json_path.name}, skipping.")
        return None

    # Collect all judge IDs, parse round data
    rounds = []
    all_judges = set()
    for rnd_str, rnd_info in data["round_judgments"].items():
        jdict = rnd_info.get("judgments", {})
        rounds.append(jdict)
        all_judges.update(jdict.keys())

    all_judges = sorted(all_judges)
    if not rounds:
        print(f"No rounds in {json_path.name}, skipping.")
        return None

    # Determine the unique final winner across all rounds
    model_round_wins = Counter()
    for jdict in rounds:
        w = get_round_winner_all_judges(jdict)
        if w and w != "tie":
            model_round_wins[w] += 1
    if not model_round_wins:
        print(f"{json_path.name}: All rounds tie or no votes => no single final winner.")
        return None

    sorted_winners = model_round_wins.most_common()
    top_model, top_count = sorted_winners[0]
    # Check tie at top
    if len(sorted_winners) > 1 and sorted_winners[1][1] == top_count:
        print(f"{json_path.name}: There's a tie for final winner => no single final winner.")
        return None

    final_winner = top_model
    total_judges = len(all_judges)

    # We'll gather subset_size -> fraction from this file
    results_for_file = {}

    candidate_sizes = [1, 2, 3,4, 5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16]
    valid_sizes = [s for s in candidate_sizes if s <= total_judges]

    for n in valid_sizes:
        # Do 10 random subsets
        fractions = []
        for _ in range(NUM_SUBSAMPLES):
            chosen_judges = random.sample(all_judges, n)
            fw_wins = 0
            decided = 0
            for jdict in rounds:
                sw = get_subset_round_winner(jdict, chosen_judges)
                if sw and sw != "tie":
                    decided += 1
                    if sw == final_winner:
                        fw_wins += 1
            frac = fw_wins / decided if decided > 0 else 0.0
            fractions.append(frac)

        avg_frac = statistics.mean(fractions)
        results_for_file[n] = avg_frac

    return results_for_file

def main(folder):
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise SystemExit(f"Error: '{folder}' is not a valid directory.")

    # Gather subset results across all files
    all_files_results = []  # each element is a dict: {subset_size -> fraction} for that file

    json_files = sorted(folder_path.glob("debate_*.json"))
    if not json_files:
        print(f"No debate_*.json files found in {folder_path}")
        return

    for f in json_files:
        file_res = analyze_single_file(f)
        if file_res is not None:
            all_files_results.append(file_res)

    if not all_files_results:
        print("No files with a unique final winner => no aggregated plot.")
        return

    # We need to combine results by subset size
    # Not all files might have the same subset sizes, but typically they do,
    # except if some file has fewer than 3 judges, etc.
    # We'll gather a list of subset_sizes from all results, then average.
    subset_sizes = sorted({ sz for d in all_files_results for sz in d.keys() })

    # For each subset size, gather fractions across all files
    aggregated = []
    for sz in subset_sizes:
        fractions_for_sz = []
        for d in all_files_results:
            if sz in d:
                fractions_for_sz.append(d[sz])
        if fractions_for_sz:
            avg_frac = statistics.mean(fractions_for_sz)
            if len(fractions_for_sz) > 1:
                std_frac = statistics.pstdev(fractions_for_sz)
            else:
                std_frac = 0.0
            aggregated.append((sz, avg_frac, std_frac))

    if not aggregated:
        print("No valid subset sizes found across files => cannot plot.")
        return

    # Print table of aggregated results
    print("\n=== Aggregated results across all files in folder ===")
    print("Subset Size | Avg Final-Winner% (across files) | StdDev")
    print("-----------------------------------------------------")
    for (sz, avg_f, std_f) in aggregated:
        print(f"{sz:^11} | {avg_f*100:27.2f} | {std_f*100:7.2f}")

    # Now produce one plot: subset size on x, average fraction on y
    x_vals = [x[0] for x in aggregated]
    y_vals = [x[1] for x in aggregated]
    y_errs = [x[2] for x in aggregated]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.5,4))
    plt.errorbar(
        x_vals, 
        y_vals, 
        yerr=y_errs, 
        fmt='-o', 
        capsize=4, 
        label="Avg Final-Winner Fraction"
    )
    plt.xlabel("Number of Judges in Subset")
    plt.ylabel("Win-Rate for Each File's Final Winner")
    plt.title(f"Aggregated Over {len(all_files_results)} Files")
    plt.ylim(0,1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_png = folder_path / "folder_plot.png"
    plt.savefig(out_png)
    plt.close()
    print(f"\nPlot saved to {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate final-winner fractions across all debate_*.json in a folder and produce ONE plot."
    )
    parser.add_argument("folder", help="Path to folder containing debate_*.json")
    args = parser.parse_args()

    main(args.folder)