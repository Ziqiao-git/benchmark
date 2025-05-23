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
    "openrouter-google-gemma-2-9b-it",
    # 15. microsoft/phi-4-reasoning-plus
    "openrouter-phi-4-reasoning-plus",
    #16. QwQ-32B
    "openrouter-QwQ-32B",
]
pairs = list(combinations(all_models, 2))

topic="Questioner is a client of the responder, who propose a very challenging, sophisticated real-world need that need to be addressed, the question should be highly specific scenario-based that uncovers the core of their problem—nothing generic or trivial. The responder need to give a valid profession solution to questioner's need, and the solution should be a valid profession solution",


detailed_instruction_sets = [
# ["How do you approach risk assessment and quality control? Can you provide examples of how you’ve ensured the safety and reliability of biomedical devices?"],
# ["Please explain any ethical boundaries that you don’t believe should be crossed in biomedical engineering."],
# ["Can you differentiate between DNA fingerprinting and therapeutic cloning?"],
# ["Can you explain MRI technology and how to use it?"],
# ["Can you describe myoelectric control and how to apply it in biomedical engineering?"],
#ee
    ["When a critical component fails unexpectedly, impacting the overall system performance, how would you troubleshoot the issue and implement a quick solution to minimize downtime?"],
#     ["During a project, you may encounter unexpected electromagnetic interference that affects sensitive equipment. How would you approach this issue to maintain the electrical system’s integrity?"],
# ["You are tasked with integrating a new electrical subsystem into an existing complex system. How would you ensure a seamless integration process and minimize potential disruptions to the overall system?"],
# ["A communication line uses a shield twisted pair (STP) cable that has a velocity factor of  0.7.What is most nearly the wavelength of a  2GHz signal traveling on the STP cable?"],
# ["The lobby of a hotel is 200ft (W)×100ft (L)×30 ft (H). Target illumination level for the lobby is 50fc. The selected luminaire for this application has 8 lamps and each lamp has an output of 750 lumens. The luminaires will be mounted at a height of 25 ft from the floor and working plane will be 3 ft from the floor. Lamp lumen depreciation factor is 0.90, luminaire dirt depreciation factor is 0.85, coefficient of utilization is 0.80, wall luminance coefficient is 0.25, ceiling cavity luminance coefficient is 0.30 and room position multiplier is 0.75. The minimum number of luminaires that will be required for this application are _____."],
# medical1
#     ["What steps do you take to ensure customer prescriptions don't interact negatively with other medications they may take?"],
# ["How do you stay updated on new medications and pharmacy practices?"],
# ["How do you maintain patient confidentiality when preparing and dispensing medications?"],
# ["Can you discuss your experience with handling medication recalls and ensuring patient safety during such situations?"],
# ["How do you ensure accuracy in medication dispensing and prescription processing?"],
#mental1
# ["How do you maintain boundaries with clients during a counseling session?"],
# ["How do you use nonverbal cues during a counseling session?"],
# ["A client presents with long-term depression and struggles with completing tasks and maintaining relationships. What interventions would you use to help this client increase their motivation and improve their sense of self-worth?"],
# ["Describe a time when you had to work with a patient who had a highly complex mental health issue. What task did you have to take on in this situation, what actions did you take, and what was the result?"],
# ["A client has a history of substance abuse and is struggling with maintaining sobriety. How would you develop a treatment plan to support this client’s recovery and prevent relapse?"],
    # mental 2
#     ["What types of assessments, specific to trauma or evidence of it, have you used to help your clients?"],
# ["Can you tell me what DSM is and how therapists use it in practice?"],
# ["What methods do you use with your clients to help them avoid secondary trauma?"],
# ["What is the relationship between psychiatrists and therapists?"],
# ["What would you consider to be the leading causes of mental health issues today?"],

]

# -------------- Helper: run *one* debate spec -----------------------------
async def run_single_debate(model_a_id, model_b_id, rounds=1):
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
        RESULTS_DIR = f"Prompt_test_{idx}_9R_16J_parallel_debate_results"
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Fresh random judges for this batch
        judge_ids = all_models
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