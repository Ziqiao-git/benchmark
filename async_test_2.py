# parallel_debates.py
import asyncio, itertools, random
from itertools import combinations
import os, json
import traceback
from model_interactions import ModelParticipant
from async_orchestration import AsyncDebate_and_Judge


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
pairs = list(combinations(all_models, 2))

topic="Question that is similar/related to the one in the given instruction, but it should be more interesting and challenging",

MAX_RETRIES = 5      # how many times to retry a failed debate
detailed_instruction_sets = [

    ["Suppose you are a mathematician and poet. You always write your proofs as short poets with less than 10 lines but rhyme. Prove the square root of 2 is irrational number.", "Prove the Pythagorean theorem."],
    ["Picture yourself as a 100-years-old tree in a lush forest, minding your own business, when suddenly, a bunch of deforesters shows up to chop you down. How do you feel when those guys start hacking away at you?", "Come up with a proposal to convince the deforesters to stop cutting you down and other trees."],
    ["When rolling two dice, what is the probability that you roll a total number that is at least 3?", "Continue from previous question. What's the probability that you roll a number which is even or at least 3?"],
    ["The vertices of a triangle are at points (0, 0), (-1, 1), and (3, 3). What is the area of the triangle?", "What's area of the circle circumscribing the triangle?"],
    ["Thomas is very healthy, but he has to go to the hospital every day. What could be the reasons?", "Can you explain why the above question is interesting?"],
    ["Describe a vivid and unique character, using strong imagery and creative language. Please answer in fewer than two paragraphs.", "Revise your previous response and incorporate an allusion to a famous work of literature or historical event in each sentence."],
    ["Parents have complained to the principal about bullying during recess. The principal wants to quickly resolve this, instructing recess aides to be vigilant. Which situation should the aides report to the principal?\na) An unengaged girl is sitting alone on a bench, engrossed in a book and showing no interaction with her peers.\nb) Two boys engaged in a one-on-one basketball game are involved in a heated argument regarding the last scored basket.\nc) A group of four girls has surrounded another girl and appears to have taken possession of her backpack.\nd) Three boys are huddled over a handheld video game, which is against the rules and not permitted on school grounds.", "If the aides confront the group of girls from situation (c) and they deny bullying, stating that they were merely playing a game, what specific evidence should the aides look for to determine if this is a likely truth or a cover-up for bullying?"],
    ["Which word does not belong with the others?\ntyre, steering wheel, car, engine", "Could you replace it with a word that belongs with the others?"],
    ["Given that f(x) = 4x^3 - 9x - 14, find the value of f(2).", "Find x such that f(x) = 0."]
]

# -------------- Helper: run *one* debate spec -----------------------------
async def run_single_debate(model_a_id, model_b_id, rounds=3):
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

    return dict(results)  
if __name__ == "__main__":
    for idx, detailed_instructions in enumerate(detailed_instruction_sets, 1):
        # Overwrite the globals that run_single_debate & main rely on
        instruction_set = [topic, detailed_instructions]
        RESULTS_DIR = f"MT_{idx}_parallel_debate_results"
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Fresh random judges for this batch
        judge_ids = random.sample(all_models, 5)
        judges = [ModelParticipant(j, role="judge") for j in judge_ids]

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