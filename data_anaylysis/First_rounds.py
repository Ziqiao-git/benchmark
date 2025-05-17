#!/usr/bin/env python3
"""
first_rounds_aggregated.py

1) For each debate_*.json file in <results_folder>:
   - Identify how many total rounds (R).
   - Determine the file's final winner (the model that wins the most total rounds).
   - For subset sizes in [1,2,3,4,5,6,7,8,9,11,13,15] (up to R):
     - Take the *first* N rounds (0..N-1).
     - If N is even and the prefix winner is "tie," skip this data point (omit from aggregates).
     - "is_match" = does prefix winner == the overall final winner?
     - "fw_fraction" = fraction of those N rounds won by the final winner.
   - Collect these results per file.

2) Aggregate across ALL files:
   - For each subset size, gather all data points from all files (unless skipped).
   - Compute the average match rate (fraction of data points that have is_match=True)
     and the average final-winner fraction.

3) Print a summary table and create a single plot "round_plot.png":
   - x-axis = subset size
   - y-axis = [0..1]
   - two lines:
       a) average match rate
       b) average final-winner fraction
   - saved in <results_folder>.

Usage:
  python first_rounds_aggregated.py <results_folder>

Requires:
  pip install matplotlib (for plotting)
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # so it can run without an interactive display
import matplotlib.pyplot as plt

def analyze_single_file(json_path):
    """
    Returns a list of tuples (subset_size, is_match, fw_fraction), or None if skipping the file.
    
    Logic:
      1) Determine overall final winner by counting wins across *all* rounds.
      2) For each candidate subset size <= total_rounds:
         - Get the prefix of that length.
         - Find which model is the "prefix winner" (if exactly one top label).
         - If the subset size is even AND the prefix winner is 'tie', SKIP that data point.
         - Otherwise record:
             is_match: (prefix_winner == final_winner?)
             fw_fraction: fraction of prefix rounds that final_winner took
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Skipping {json_path.name} (parse error: {e})")
        return None

    if "round_judgments" not in data:
        print(f"No round_judgments in {json_path.name}; skipping.")
        return None

    rj = data["round_judgments"]
    # Sort round keys numerically
    round_nums = sorted(rj.keys(), key=lambda x: int(x))

    all_round_winners = []
    for rnd_str in round_nums:
        w = rj[rnd_str].get("winner")
        if w and w != "tie":
            all_round_winners.append(w)
        else:
            all_round_winners.append("tie")

    total_rounds = len(all_round_winners)
    if total_rounds == 0:
        print(f"{json_path.name}: zero rounds => skip.")
        return None

    # Determine the file's overall final winner (unique top wins across *all* rounds)
    c = Counter(all_round_winners)
    # remove 'tie' from the tally
    if 'tie' in c:
        del c['tie']
    if not c:
        print(f"{json_path.name}: all rounds are tie => no single final winner.")
        return None

    sorted_c = c.most_common()
    top_model, top_count = sorted_c[0]
    # tie for top?
    if len(sorted_c) > 1 and sorted_c[1][1] == top_count:
        print(f"{json_path.name}: there's a tie for final winner => skip.")
        return None

    final_winner = top_model

    # candidate sizes, including 1,2 and so on
    candidate_sizes = [1,2,3,4,5,6,7,8,9,11,13,15]
    subset_sizes = [sz for sz in candidate_sizes if sz <= total_rounds]

    results = []
    for sz in subset_sizes:
        prefix = all_round_winners[:sz]
        prefix_counter = Counter(prefix)
        # remove 'tie'
        if 'tie' in prefix_counter:
            del prefix_counter['tie']

        if not prefix_counter:
            # means the entire prefix is tie
            subset_winner = "tie"
        else:
            sc_sorted = prefix_counter.most_common()
            w_label, w_count = sc_sorted[0]
            # Check if there's a tie for top
            if len(sc_sorted) > 1 and sc_sorted[1][1] == w_count:
                subset_winner = 'tie'
            else:
                subset_winner = w_label

        # If even size and subset winner is tie, skip
        if (sz % 2 == 0) and (subset_winner == 'tie'):
            continue

        # is_match?
        is_match = (subset_winner == final_winner)

        # final_winner fraction in the prefix
        fw_count = prefix_counter.get(final_winner, 0)
        fw_fraction = fw_count / sz

        results.append((sz, is_match, fw_fraction))

    return results

def main(folder):
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise SystemExit(f"Error: '{folder}' is not a valid directory.")

    json_files = sorted(folder_path.glob("debate_*.json"))
    if not json_files:
        print(f"No debate_*.json found in {folder_path}")
        return

    all_file_results = []
    for f in json_files:
        r = analyze_single_file(f)
        if r is not None:
            all_file_results.append(r)

    if not all_file_results:
        print("No valid files => no aggregation or plot.")
        return

    # all_file_results is a list of per-file lists of (subset_size, is_match, fw_fraction).
    # We'll gather them by subset_size into aggregated_dict[subset_size] = list of (match, fw_frac)
    from collections import defaultdict
    aggregated_dict = defaultdict(list)
    for file_res in all_file_results:
        for (sz, is_match, fw_frac) in file_res:
            aggregated_dict[sz].append((is_match, fw_frac))

    # Now compute aggregated stats for each subset_size
    aggregated_stats = []
    for sz in sorted(aggregated_dict.keys()):
        values = aggregated_dict[sz]  # list of (bool, float)
        n_files = len(values)
        if n_files == 0:
            continue
        match_rate = sum(1 for (m, _) in values if m) / n_files
        avg_fw_fraction = sum(fw for (_, fw) in values) / n_files
        aggregated_stats.append((sz, n_files, match_rate, avg_fw_fraction))

    if not aggregated_stats:
        print("No data to plot after even-size tie skipping => no results.")
        return

    print("\n=== Aggregated Results Across All Files ===")
    print("SubsetSize | #Files |  MatchRate  |  FinalWinnerFrac ")
    print("-----------------------------------------------------")
    for (sz, n_f, mr, fwf) in aggregated_stats:
        print(f"{sz:^10} | {n_f:^6} | {mr*100:10.2f}% | {fwf*100:17.2f}%")

    # Plot 
    x_vals = [x[0] for x in aggregated_stats]
    file_counts = [x[1] for x in aggregated_stats]
    match_rates = [x[2] for x in aggregated_stats]
    fw_fracs = [x[3] for x in aggregated_stats]

    plt.figure(figsize=(6,4))
    plt.plot(x_vals, match_rates, marker='o', linestyle='-', color='blue', label='Match Rate')
    plt.plot(x_vals, fw_fracs,   marker='o', linestyle='-', color='red',  label='FinalWinner Fraction')
    # If you also want to see # of files that contributed each point, you can do a second y-axis or so.

    plt.xlabel("Number of Rounds (Prefix Size)")
    plt.ylabel("Rate / Fraction")
    plt.ylim(0,1)
    plt.title(f"Aggregated across {len(all_file_results)} total files")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_file = folder_path / "round_plot.png"
    plt.savefig(out_file)
    plt.close()
    print(f"\nPlot saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate 'first N rounds' results across all files. If N is even and prefix is tie, skip that data point."
    )
    parser.add_argument("folder", help="Folder containing debate_*.json files")
    args = parser.parse_args()

    main(args.folder)