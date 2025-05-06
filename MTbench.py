from Debate_orchestration import debate_orchestration
from Judge_orchestration import judge_orchestration
from model_interactions import ModelParticipant, Debate
from tqdm import tqdm
import os
import json

topic = "Question that is similar/related to the one in the given instruction"
detailed_instructions = [
    [
        "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
        "Rewrite your previous response. Start every sentence with the letter A."
    ],
    # [
    #     "Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. Keep the email short and to the point.",
    #     "Take a moment to evaluate and critique your own response."
    # ],
    # [
    #     "Imagine you are writing a blog post comparing two popular smartphone models. Develop an outline for the blog post, including key points and subheadings to effectively compare and contrast the features, performance, and user experience of the two models. Please answer in fewer than 200 words.",
    #     "Take your previous response and rephrase it as a limerick."
    # ]
    # ...
    # For now, only 3 sets are shown. 
    # You can re-add the rest or keep it short for demonstration.
]

# Create model list
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

num_instructions = len(detailed_instructions)
num_models = len(all_models)
pairs_count = num_models * (num_models - 1) // 2
total_debates = pairs_count * num_instructions

progress = tqdm(total=total_debates, desc="Debate Progress")

for instr_idx, instruction in enumerate(detailed_instructions, start=1):
    project_name = f"MTbench_{instr_idx}"
    os.makedirs(f"{project_name}/debates", exist_ok=True)
    
    for i in range(num_models):
        for j in range(i + 1, num_models):
            model_a_id = all_models[i]
            model_b_id = all_models[j]
            
            # Path for the debate results
            results_path = f"{project_name}/debates/debate_results_{model_a_id}_{model_b_id}.json"
            
            # =========== RESUME CHECK =============
            if os.path.exists(results_path):
                # If the file exists, let's try to load it
                try:
                    with open(results_path, "r") as f:
                        existing_data = json.load(f)
                    
                    # Minimal check: does it have "results" -> "transcript"?
                    if "results" in existing_data and "transcript" in existing_data["results"]:
                        # We'll skip if it looks valid
                        # If you want a stricter check, 
                        # verify the transcript is non-empty, etc.
                        print(f"Skipping {model_a_id} vs {model_b_id} (already done for instruction set {instr_idx})")
                        progress.update(1)
                        continue
                except:
                    pass
                # If the file is corrupted or incomplete, we re-run
                
            # =========== RUN THE DEBATE =============
            print(f"Running debate between {model_a_id} and {model_b_id} for instruction set {instr_idx}")
            model_a = ModelParticipant(model_a_id, role="debater")
            model_b = ModelParticipant(model_b_id, role="debater")

            # Single-round debate
            debate_orchestration(
                [model_a, model_b],
                topic=topic,
                rounds=1,
                project_name=project_name,
                detailed_instructions=instruction
            )

            progress.update(1)

progress.close()
print("All debates done! Now you can run the judge step (or the next steps).")
