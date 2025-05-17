#!/usr/bin/env python3
"""
per_round_stats.py

Compute round-level win/loss/tie stats for *one* model against all rivals.
The script looks only at `round_judgments.*.winner` in each JSON file.
"""

import argparse, json, os, re
import math    
from collections import defaultdict, Counter
from pathlib import Path


PAIR_RE = re.compile(r"^debate_(.+?)_vs_(.+?)\.json$")
# --- helper ---------------------------------------------------------------
def bernoulli_cross_entropy(p: float, q: float = 0.5, eps: float = 1e-9) -> float:
    """
    Cross-entropy H(P,Q) for two Bernoulli distributions (bits).
    p – empirical win-prob; q – reference prob (default 0.5).
    """
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    h_nats = -(p * math.log(q) + (1 - p) * math.log(1 - q))
    return h_nats / math.log(2)           # convert nats → bits

def load_round_winners(path: Path):
    """Return {round_num: 'model_id'|'tie'} or None on error."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return {int(rn): res["winner"]
                for rn, res in data["round_judgments"].items()}
    except Exception:
        return None  # skip unreadable / incomplete files

def main(folder: str, target: str) -> None:
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    # stats[target][rival] = Counter({'win':x,'lose':y,'tie':z})
    stats = defaultdict(lambda: Counter(win=0, lose=0, tie=0))

    for file in folder.iterdir():
        m = PAIR_RE.match(file.name)
        if not (file.is_file() and m):
            continue

        model_a, model_b = m.groups()
        if target not in (model_a, model_b):
            continue  # not involving our model

        winners = load_round_winners(file)
        if winners is None:
            continue

        rival = model_b if target == model_a else model_a
        for _, w in winners.items():
            if w == "tie":
                stats[rival]["tie"] += 1
            elif w == target:
                stats[rival]["win"] += 1
            elif w == rival:
                stats[rival]["lose"] += 1
            # else: malformed winner string, ignore

    if not stats:
        print(f"No debates found for model {target}")
        return

    # ---- pretty print -----------------------------------------------------
    print(f"\nRound-level results for: {target}\n")
    header = f"{'Rival':<45} | Wins | Losses | Ties | Win-rate"
    print(header)
    print("-" * len(header))

    for rival, ctr in sorted(stats.items()):
        w, l, t = ctr["win"], ctr["lose"], ctr["tie"]
        decided = w + l
        winrate = f"{w/decided:.2%}" if decided else "–"
        print(f"{rival:<45} | {w:^4} | {l:^6} | {t:^4} | {winrate:>8}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-round win/loss/tie stats for one model.")
    parser.add_argument("results_folder",
                        help="Path to folder with debate_*.json files")
    parser.add_argument("model_id",
                        help="Model ID to evaluate (exact string in filenames)")
    args = parser.parse_args()
    main(args.results_folder, args.model_id)