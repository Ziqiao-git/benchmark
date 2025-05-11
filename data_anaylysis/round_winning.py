#!/usr/bin/env python3
"""
analyze_debate_results.py

Summarize round‑level win/loss/tie stats for ONE target model
against every rival and compute the Bernoulli cross‑entropy
between each empirical win‑rate and a reference probability q.

* Looks ONLY at `round_judgments.*.winner` in every debate_*.json
* Ignores final_assessment
* Skips corrupted / missing files

Usage
-----
python analyze_debate_results.py <results_folder> <model_id> [--baseline 0.5] [--csv]

• <results_folder>  directory that contains debate_<A>_vs_<B>.json files
• <model_id>        the model you want to evaluate (must appear in filenames)
• --baseline q      reference win probability for cross‑entropy (default 0.5)
• --csv             also write summary.csv inside <results_folder>
"""

import argparse, json, os, re, math, csv
from collections import defaultdict, Counter
from pathlib import Path

PAIR_RE = re.compile(r"^debate_(.+?)_vs_(.+?)\.json$")


# ---------------------------------------------------------------------------#
# Helpers
# ---------------------------------------------------------------------------#
def bernoulli_cross_entropy(p: float, q: float = 0.5, eps: float = 1e-9) -> float:
    """H(P,Q) for Bernoulli distributions (log2 bits)."""
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    h_nats = -(p * math.log(q) + (1 - p) * math.log(1 - q))
    return h_nats / math.log(2)  # convert nats ➜ bits


def load_round_winners(path: Path):
    """Return {round_num: winner} or None on error."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return {int(rn): res["winner"] for rn, res in data["round_judgments"].items()}
    except Exception:
        return None


# ---------------------------------------------------------------------------#
def main(folder: str, target: str, baseline: float = 0.5, csv_out: bool = False) -> None:
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    stats = defaultdict(lambda: Counter(win=0, lose=0, tie=0))

    for file in folder_path.iterdir():
        m = PAIR_RE.match(file.name)
        if not (file.is_file() and m):
            continue

        model_a, model_b = m.groups()
        if target not in (model_a, model_b):
            continue

        winners = load_round_winners(file)
        if winners is None:
            continue

        rival = model_b if target == model_a else model_a
        for w in winners.values():
            if w == "tie":
                stats[rival]["tie"] += 1
            elif w == target:
                stats[rival]["win"] += 1
            elif w == rival:
                stats[rival]["lose"] += 1

    if not stats:
        print(f"No debates found for model '{target}'.")
        return

    # ------------  Print table  --------------------------------------------
    header = f"{'Rival':<45} | Wins | Losses | Ties | Win‑rate | CE(bits)"
    sep = "-" * len(header)
    print(f"\nRound‑level results for: {target}\n")
    print(header)
    print(sep)

    rows = []
    for rival, ctr in sorted(stats.items()):
        w, l, t = ctr["win"], ctr["lose"], ctr["tie"]
        decided = w + l
        if decided == 0:
            winrate = "–"
            ce_str = "–"
            ce_val = None
        else:
            p = w / decided
            winrate = f"{p:.2%}"
            ce_val = bernoulli_cross_entropy(p, baseline)
            ce_str = f"{ce_val:.3f}"
        print(f"{rival:<45} | {w:^4} | {l:^6} | {t:^4} | {winrate:>8} | {ce_str:>7}")
        rows.append((rival, w, l, t, winrate, ce_val))

    # ------------  Optional CSV  -------------------------------------------
    if csv_out:
        csv_path = folder_path / "summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rival", "wins", "losses", "ties", "win_rate", "cross_entropy_bits"])
            for r, w, l, t, wr, ce in rows:
                writer.writerow([r, w, l, t, wr, f"{ce:.6f}" if ce is not None else ""])
        print(f"\nCSV written to {csv_path}")


# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round‑level stats + cross‑entropy")
    parser.add_argument("results_folder", help="Folder containing debate_*.json files")
    parser.add_argument("model_id", help="Model ID to evaluate")
    parser.add_argument("--baseline", type=float, default=0.5,
                        help="Reference win probability q (default 0.5)")
    parser.add_argument("--csv", action="store_true", help="Write summary.csv")
    args = parser.parse_args()

    main(args.results_folder, args.model_id, args.baseline, args.csv)