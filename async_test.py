# simple_async_debate.py
import asyncio
import json
import time
from model_interactions import ModelParticipant      # your wrapper class
from async_orchestration import AsyncDebate_and_Judge  # your orchestrator
import random
async def main() -> None:
    start_time = time.time()
    # --- 1. Create participants -------------------------------------------
    debater_a = ModelParticipant(model_id="o1",       role="debater")
    debater_b = ModelParticipant(model_id="o4-mini",  role="debater")
    

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
    judge_list = random.sample(all_models, 5)
    judges = []
    for judge in judge_list:
        judges.append(ModelParticipant(model_id=judge, role="judge"))
    topic="Question that is similar/related to the one in the given instruction ",
    detailed_instructions=[
        "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
        "Rewrite your previous response. Start every sentence with the letter A."
    ]
    # --- 2. Configure the debate ------------------------------------------
    debate = AsyncDebate_and_Judge(
        participants=[debater_a, debater_b],  
        rounds=5,                             
        judges_list=judges,                 
        instruction_set=[topic, detailed_instructions]
    )

    # --- 3. Run the debate --------------------------------------------------
    results = await debate.run_debate()

    # --- 4. Print or persist the outcome -----------------------------------
    #    `results` is whatever your orchestrator returns (often a dict).
        # --- 4. Persist the outcome to a JSON file -----------------------------
    with open("debate_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Optional: also pretty‑print to the console
    end_time = time.time()
    print(f"Debate with results completed in {end_time - start_time:.2f} seconds")
if __name__ == "__main__":
    asyncio.run(main())