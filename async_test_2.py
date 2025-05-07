# parallel_debates.py
import asyncio, itertools, random
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
pairs = [
    ("o1", "o3"),
    ("o1", "o4-mini"),
    ("o3", "o4-mini"),
    ("openrouter-Gemini-2.5-pro", "openrouter-QwQ-32B"),
    ("openrouter-Grok-3-Beta", "openrouter-Qwen3-235B-A22B"),
    ("deepseek", "openrouter-Amazon_Nova_1"),
    ("openrouter-Gemini-2.5-flash-thinking", "openrouter-deepseek-v3-0324"),
    ("openrouter-claude-3.7-sonnet-thinking", "o1"),
    ("o3", "openrouter-Gemini-2.5-flash-thinking"),
    ("o4-mini", "openrouter-claude-3.7-sonnet-thinking"),
]

topic = "General knowledge"
detail = "Answer concisely in ≤40 words."
instruction_set = [topic, detail]

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
async def main(max_concurrent=5):
    sem = asyncio.Semaphore(max_concurrent)

    async def run_with_sem(pair):
        async with sem:
            a, b = pair
            print(f"▶️  Starting debate {a} vs {b}")
            try:
                res = await run_single_debate(a, b)
                print(f"✅ Finished debate {a} vs {b}")
            except Exception as e:
                # Capture exception so the rest of the batch keeps running
                print(f"❌ Debate {a} vs {b} failed: {e}")
                res = {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "model_a": a,
                    "model_b": b,
                }
            # Persist success *or* failure to a JSON file
            fname = f"debate_{a}_vs_{b}.json".replace('/', '_')
            with open(os.path.join(RESULTS_DIR, fname), "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            return (pair, res)

    tasks = [asyncio.create_task(run_with_sem(p)) for p in pairs]
    results = await asyncio.gather(*tasks)
    return dict(results)   # { (modelA,modelB): result_dict }

# -------------- Kick it off ------------------------------------------------
if __name__ == "__main__":
    all_results = asyncio.run(main())
    # Save a combined summary file
    with open(os.path.join(RESULTS_DIR, "all_debates_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\n=== All debates complete! ===")
    for k, v in all_results.items():
        print(k, "→ winner:", v["final_assessment"]["overall_winner"])