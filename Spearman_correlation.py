import json
import os
from scipy.stats import spearmanr

all_models = [
    "openrouter-Gemini-2.5-pro",       # 1. Gemini-2.5-pro
    "o3",                              # 2. GPT
    "openrouter-Grok-3-Beta",         # 3. Grok-3-Beta
    "openrouter-Gemini-2.5-flash-thinking",  # 4. Gemini-2.5-Flash-thinking
    "openrouter-deepseek-v3-0324",    # 5. deepseek-v3-0324
    "deepseek",                       # 6. Deepseek-R1
    "o4-mini",                        # 7. GPT
    "o1",                             # 8. GPT
    "openrouter-Qwen3-235B-A22B",     # 9. Qwen3-235B-A22B
    "openrouter-QwQ-32B",             # 10. QwQ-32B
    "openrouter-claude-3.7-sonnet-thinking",  # 11. Claude-3.7-Sonnet-thinking
    "openrouter-Amazon_Nova_1"        # 12. Amazon:Nova Pro 1.0
]


# We'll store a 1-based index for each model (Spearman correlation typically uses ranks).
original_rank_map = {model: (i+1) for i, model in enumerate(all_models)}
base_dir = os.getcwd() 
NUM_FOLDERS = 10

for i in range(1, NUM_FOLDERS +2):
    folder_name = f"MT_{i}_parallel_debate_results"
    folder_path = os.path.join(base_dir, folder_name)
    if i == NUM_FOLDERS + 1:
        JSON_FILE_PATH = "final_elo_scores.json"  
    else:
        JSON_FILE_PATH = os.path.join(folder_path, f"folder_elo_scores.json")  # Change as needed

    # Read JSON content
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # We expect data["ranking"] to be a list of objects like:
    #   {
    #     "rank": 1,
    #     "model": "o3",
    #     "elo": 1332.52
    #   }
    ranking_list = data.get("ranking", [])

    # Build a {model -> final_rank} dict from the JSON
    final_rank_map = {}
    for entry in ranking_list:
        model_id = entry["model"]
        final_rank = entry["rank"]  # 1-based rank from Elo
        final_rank_map[model_id] = final_rank

    # 3) We only compute correlation for models that appear in both sets.
    common_models = set(original_rank_map.keys()) & set(final_rank_map.keys())

    if len(common_models) < 2:
        print("Not enough overlapping models to compute Spearman correlation.")
    else:
        # Build arrays of original ranks vs. final Elo ranks
        original_positions = []
        final_elo_ranks = []
        for m in common_models:
            original_positions.append(original_rank_map[m])
            final_elo_ranks.append(final_rank_map[m])

        # 4) Compute Spearman correlation
        correlation, p_value = spearmanr(original_positions, final_elo_ranks)

        # 5) Print or store the results
        print(f"Compare to Chatbot Arena Overall Ranking")
        if i == NUM_FOLDERS + 1:
            print(f"Using the all files")
        else:
            print(f"Using the {folder_name} folder")
        print(f"Spearman correlation: {correlation:.3f}")
        print(f"P-value: {p_value:.3g}")
