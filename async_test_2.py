# parallel_debates.py
import asyncio, itertools, random
from itertools import combinations
import os, json
import traceback
from model_interactions import ModelParticipant
from async_orchestration import AsyncDebate_and_Judge

# ----- Prepare 10 debate specs (pick your own pairs) -----------------------
all_models = [
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
# ---- generate every unique unordered pair (66 total) --------------------
pairs = list(combinations(all_models, 2))

topic="Question that is similar/related to the one in the given instruction ",
detailed_instructions=[
    "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
    "Rewrite your previous response. Start every sentence with the letter A."
]
instruction_set = [topic, detailed_instructions]

MAX_RETRIES = 5      # how many times to retry a failed debate

# Directory to store JSON outputs for each debate
RESULTS_DIR = "parallel_debate_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# A single list of shared judges (or pick per-debate)
judge_ids = ["openrouter-Grok-3-Beta"]
judges = [ModelParticipant(j, role="judge") for j in judge_ids]

# -------------- Helper: run *one* debate spec -----------------------------
async def run_single_debate(model_a_id, model_b_id, rounds=2):
    debaters = [
        ModelParticipant(model_a_id, role="debater"),
        ModelParticipant(model_b_id, role="debater")
    ]
    adj = AsyncDebate_and_Judge(
        participants=debaters,
        judges_list=judges,
        instruction_set=instruction_set,
        rounds=rounds,
    )
    return await adj.run_debate()   # returns dict with transcript + scores

# -------------- Master coroutine that throttles concurrency ---------------
async def main(max_concurrent=10):
    sem = asyncio.Semaphore(max_concurrent)

    async def run_with_sem(pair):
        async with sem:
            a, b = pair
            attempt = 0
            while attempt < MAX_RETRIES:
                attempt += 1
                print(f"▶️  {a} vs {b}  (attempt {attempt}/{MAX_RETRIES})")
                try:
                    res = await run_single_debate(a, b)
                    print(f"✅  Success {a} vs {b}")
                    break
                except Exception as e:
                    print(f"❌  Attempt {attempt} for {a} vs {b} failed: {e}")
                    if attempt == MAX_RETRIES:
                        res = {
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                            "model_a": a,
                            "model_b": b,
                            "attempts": attempt,
                        }
                        print(f"🛑  Giving up on {a} vs {b}")
            # Persist result (success or failure)
            fname = f"debate_{a}_vs_{b}.json".replace('/', '_')
            with open(os.path.join(RESULTS_DIR, fname), "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            return (pair, res)

    tasks = [asyncio.create_task(run_with_sem(p)) for p in pairs]
    results = await asyncio.gather(*tasks)

    failed_pairs = [pair for pair, res in results if isinstance(res, dict) and res.get("error")]
    if failed_pairs:
        with open(os.path.join(RESULTS_DIR, "state.json"), "w", encoding="utf-8") as f:
            json.dump({"failed_pairs": [f"{a}__{b}" for a, b in failed_pairs]}, f, indent=2, ensure_ascii=False)

    return dict(results)   # { (modelA,modelB): result_dict }

# -------------- Kick it off ------------------------------------------------
if __name__ == "__main__":
    all_results = asyncio.run(main())

    # Convert tuple keys to strings so JSON can serialize them
    serializable_results = {f"{k[0]}__{k[1]}": v for k, v in all_results.items()}

    # Save a combined summary file
    with open(os.path.join(RESULTS_DIR, "all_debates_summary.json"), "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print("\n=== All debates complete! ===")
    for k, v in all_results.items():
        print(f"{k[0]} vs {k[1]} → winner: {v.get('final_assessment', {}).get('overall_winner', 'N/A')}")