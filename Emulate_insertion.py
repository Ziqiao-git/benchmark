#!/usr/bin/env python3
"""
rank_simulated.py
──────────────────────────────────────────────────────────────────────────────
Offline replica of Ranking.py:

1.  Pick 5 base models that have complete pairwise results in JSON files.
2.  Rank those 5 by Elo (using the stored outcomes only).
3.  Insert every remaining model by binary search + 1-step bubble.
4.  Output the final ordering as JSON / TXT.

Inputs
------
Folder must contain files named either
  • evaluation_results_<A>_<B>.json
  • debate_<A>_vs_<B>.json
in which the winner can be parsed.

Usage
-----
python rank_simulated.py --folder PATH [--k 32 --seed 0]

Author: 2025-05-17
"""

import argparse, json, random, re
from pathlib import Path
from typing import Dict, Tuple, List
from collections import Counter

# ────────────────────────────────────────────────────────────────────────────
# 0.  regex & helpers
PAT_EVAL = re.compile(r"^evaluation_results_(.+?)_(.+?)\.json$")
PAT_DEB  = re.compile(r"^debate_(.+?)_vs_(.+?)\.json$")

def majority_round_winner(rj: Dict) -> str:
    """Return 'A','B','tie' (modelA/modelB/tie) from round_judgments."""
    tallies = Counter(rd.get("winner") for rd in rj.values())
    a, b = tallies.get("modelA", 0), tallies.get("modelB", 0)
    if a > b:   return "A"
    if b > a:   return "B"
    return "tie"

def parse_outcome(path: Path) -> Tuple[str,str,str]:
    """
    Extract (modelA_id, modelB_id, outcome) from one JSON file.
    outcome ∈ {'A','B','tie','uncertain'}
    """
    name = path.name
    m = PAT_EVAL.match(name) or PAT_DEB.match(name)
    if not m:
        return None
    A, B = m.groups()
    try:
        data = json.loads(path.read_text())
    except Exception:
        return A, B, "uncertain"

    # 1) evaluation file with overall_winner
    overall = (
        data.get("evaluation", {})
        .get("results", {})
        .get("overall_winner")
    )
    if overall in (A, B):
        return A, B, "A" if overall == A else "B"
    if overall == "tie":
        return A, B, "tie"

    # 2) majority of round_judgments
    rj = (
        data.get("evaluation", {})
        .get("results", {})
        .get("round_judgments")
        or data.get("round_judgments")
        or {}
    )
    if rj:
        maj = majority_round_winner(rj)
        return A, B, maj

    return A, B, "uncertain"

# ────────────────────────────────────────────────────────────────────────────
# 1.  Build outcome table {(A,B): 'A'|'B'|'tie'}
def build_outcome_table(folder: Path) -> Dict[Tuple[str,str], str]:
    out = {}
    for p in folder.iterdir():
        tup = parse_outcome(p)
        if tup is None:        # not a debate/eval file
            continue
        A, B, res = tup
        if res == "uncertain":
            continue
        key = tuple(sorted((A, B)))
        # store winner as relative to key[0] vs key[1]
        if res == "tie":
            out[key] = "tie"
        else:
            out[key] = "A" if (res == "A" and key[0] == A) or (res == "B" and key[0] == B) else "B"
    return out

def lookup(a: str, b: str, table: Dict[Tuple[str,str], str]) -> str:
    key = tuple(sorted((a, b)))
    res = table.get(key, "uncertain")
    if res == "tie" or res == "uncertain":
        return res
    return res if key[0] == a else ("A" if res == "B" else "B")

# ────────────────────────────────────────────────────────────────────────────
# 2.  Basic Elo
def expected(ra, rb): return 1 / (1 + 10 ** ((rb - ra) / 400))
def update_elos(ra, rb, score_a, k=32):
    ea = expected(ra, rb)
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))

def elo_order(players: List[str], table, k=32, passes=5, seed=0):
    elo = {m: 1500 for m in players}
    rng = random.Random(seed)
    matches = [
        (a, b, lookup(a, b, table))
        for i, a in enumerate(players)
        for b in players[i + 1 :]
        if lookup(a, b, table) != "uncertain"
    ]
    for _ in range(passes):
        rng.shuffle(matches)
        for a, b, res in matches:
            s = 1.0 if res == "A" else 0.0 if res == "B" else 0.5
            elo[a], elo[b] = update_elos(elo[a], elo[b], s, k=k)
    return sorted(players, key=elo.get, reverse=True)

# ────────────────────────────────────────────────────────────────────────────
# 3.  Choose a fully connected base set
def find_base(models, table, n=5, rng=None):
    rng = rng or random
    rng.shuffle(models)
    for i in range(len(models) - n + 1):
        subset = models[i : i + n]
        if all(lookup(subset[x], subset[y], table) != "uncertain"
               for x in range(n) for y in range(x + 1, n)):
            return subset
    raise RuntimeError("Could not find a connected base set of size %d" % n)

# 4.  Binary search insertion using lookups
def binary_insert(model, ranked, table):
    if not ranked:
        return [model]
    lo, hi = 0, len(ranked) - 1
    pos = None
    while lo <= hi:
        mid = (lo + hi) // 2
        cmp = lookup(model, ranked[mid], table)
        if cmp == "A":          # model beats mid
            hi = mid - 1
            pos = mid
        elif cmp == "B":        # model loses to mid
            lo = mid + 1
            pos = lo
        else:                   # tie/unknown
            pos = mid + 1
            break
    ranked.insert(pos, model)

    # local bubble for safety
    idx = pos
    while idx > 0 and lookup(ranked[idx], ranked[idx - 1], table) == "A":
        ranked[idx], ranked[idx - 1] = ranked[idx - 1], ranked[idx]
        idx -= 1
    idx = pos
    while idx < len(ranked) - 1 and lookup(ranked[idx], ranked[idx + 1], table) == "B":
        ranked[idx], ranked[idx + 1] = ranked[idx + 1], ranked[idx]
        idx += 1
    return ranked

# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder with debate/eval JSON files")
    ap.add_argument("--k", type=int, default=32, help="Elo K-factor for base ordering")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        raise SystemExit("Folder not found")

    print(f"📂  Scanning {folder}")
    table = build_outcome_table(folder)
    if not table:
        raise SystemExit("No usable pairwise results found")

    models = sorted({m for pair in table for m in pair})
    rng = random.Random(args.seed)

    base = find_base(models.copy(), table, 5, rng=rng)
    print("Base set:", base)

    base_ranked = elo_order(base, table, k=args.k, seed=args.seed)
    print("Base order after Elo:", base_ranked)

    ranked = base_ranked[:]
    for m in models:
        if m in ranked:
            continue
        ranked = binary_insert(m, ranked, table)
        print(f"Inserted {m}  →  rank {ranked.index(m)+1}/{len(ranked)}")

    # write outputs
    (folder / "ranking_final.json").write_text(json.dumps(ranked, indent=2))
    with (folder / "ranking_final.txt").open("w") as f:
        for i, m in enumerate(ranked, 1):
            f.write(f"{i:3}. {m}\n")

    print("\n=== FINAL ORDER ===")
    for i, m in enumerate(ranked, 1):
        print(f"{i:3}. {m}")
    print(f"\nFiles written in {folder}: ranking_final.json / ranking_final.txt")

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()