# parallel_debates.py
import asyncio, itertools, random
from itertools import combinations
import os, json
import traceback
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from model_interactions import ModelParticipant
from async_orchestration import AsyncDebate_and_Judge


all_models = [
    "o1",
    "openrouter-QwQ-32B"
]
judges = [
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
    # 7. Qwen2.5-72B-Instruct (not in file; closest is openrouter-qwen-2-72b-instruct)
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
    "openrouter-google-gemma-2-9b-it"
]
pairs = list(combinations(all_models, 2))

topic="Questioner is a client of the responder, who propose a real life very challenging and sophisticated needs that need to be addressed, the question should not be too general or trivial or not related to the real life. The responder need to give a valid profession solution to questioner's need, and the solution should be a valid profession solution",


detailed_instruction_sets = [
["Assess, diagnose, and treat mental and emotional disorders of individuals through observation, interview, and psychological tests. Help individuals with distress or maladjustment understand their problems through their knowledge of case history, interviews with patients, and theory. Provide individual or group counseling services to assist individuals in achieving more effective personal, social, educational, and vocational development and adjustment. May design behavior modification programs and consult with medical personnel regarding the best treatment for patients."]

]

# -------------- Helper: run *one* debate spec -----------------------------
async def run_single_debate(model_a_id, model_b_id, rounds=9):
    debaters = [
        ModelParticipant(model_a_id, role="debater"),
        ModelParticipant(model_b_id, role="debater")
    ]
    adj = AsyncDebate_and_Judge(
        participants=debaters,
        judges_list=judges,
        instruction_set=instruction_set,
        rounds=rounds,
        results_dir=RESULTS_DIR,
    )
    return await adj.run_debate()   # returns dict with transcript + scores
async def main(max_concurrent=10):
    sem = asyncio.Semaphore(max_concurrent)

    # ---------------------------------------------------
    # Resume support: skip pairs that already have result files
    # ---------------------------------------------------
    completed_pairs = set()
    if os.path.isdir(RESULTS_DIR):
        for fname in os.listdir(RESULTS_DIR):
            if fname.startswith("debate_") and fname.endswith(".json"):
                core = fname[len("debate_"):-len(".json")]          # modelA_vs_modelB
                a, _, b = core.partition("_vs_")
                if a and b:
                    completed_pairs.add((a, b))
    pending_pairs = [p for p in pairs if p not in completed_pairs]
    if not pending_pairs:
        print("🔄  All pairs in this instruction set already processed.")
        return {}

    async def run_with_sem(pair):
        async with sem:
            a, b = pair
            print(f"▶️  {a} vs {b}")
            try:
                res = await run_single_debate(a, b)
                print(f"✅  Finished {a} vs {b}")
            except Exception as e:
                print(f"❌  Debate {a} vs {b} failed: {e}")
                res = {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "model_a": a,
                    "model_b": b,
                }

            fname = f"debate_{a}_vs_{b}.json".replace("/", "_")
            with open(os.path.join(RESULTS_DIR, fname), "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            return (pair, res)

    tasks = [asyncio.create_task(run_with_sem(p)) for p in pending_pairs]
    results = await tqdm_asyncio.gather(*tasks, total=len(pending_pairs), desc="Debates")

    failed_pairs = [pair for pair, res in results if isinstance(res, dict) and res.get("error")]
    if failed_pairs:
        with open(os.path.join(RESULTS_DIR, "state.json"), "w", encoding="utf-8") as f:
            json.dump({"failed_pairs": [f"{a}__{b}" for a, b in failed_pairs]}, f, indent=2, ensure_ascii=False)

    return dict(results)  
if __name__ == "__main__":
    # tqdm progress bar over the 10 detailed instruction sets
    for idx, detailed_instructions in enumerate(
        tqdm(detailed_instruction_sets, desc="Instruction sets", unit="set"), 1
    ):
        instruction_set = [topic, detailed_instructions]
        RESULTS_DIR = f"test_results_12"
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Fresh random judges for this batch
        judge_ids = judges
        judges = [ModelParticipant(j, role="judge") for j in judge_ids]
        # Do not let Nova be a judge

        print(f"\n=== Running instruction set {idx}/{len(detailed_instruction_sets)} ===")
        all_results = asyncio.run(main())

        # Convert tuple keys to strings so JSON can serialize them
        serializable_results = {f"{k[0]}__{k[1]}": v for k, v in all_results.items()}

        # Save a combined summary file
        with open(os.path.join(RESULTS_DIR, "all_debates_summary.json"), "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        # Simple console output of winners
        for (a_b, res) in all_results.items():
            a, b = a_b
            winner = res.get("final_assessment", {}).get("overall_winner", "N/A")
            print(f"{a} vs {b} → winner: {winner}")

        print(f"✔️  Finished instruction set {idx}")