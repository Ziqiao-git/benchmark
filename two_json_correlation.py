import json
from scipy.stats import spearmanr

def compute_spearman_from_json(json_data1, json_data2):
    """
    Given two dicts each containing a 'ranking' list of objects
    with {'rank': int, 'model': str}, compute Spearman correlation
    for the shared models. Any models that appear in only one JSON
    will be skipped.
    """
    # Extract {model -> rank} mappings from each JSON.
    # If the JSON has a "ranking" list with e.g. 
    #   [{ "rank": 1, "model": "A" }, { "rank": 2, "model": "B" }, ... ]
    # then rank_map_1 = { "A": 1, "B": 2, ... } and similarly for rank_map_2.
    rank_map_1 = { item["model"]: item["rank"] for item in json_data1["ranking"] }
    rank_map_2 = { item["model"]: item["rank"] for item in json_data2["ranking"] }

    # Find the intersection of models present in both
    common_models = set(rank_map_1.keys()) & set(rank_map_2.keys())

    # Build two parallel lists of ranks for the models in the intersection
    ranks_1 = []
    ranks_2 = []
    for model in common_models:
        ranks_1.append(rank_map_1[model])
        ranks_2.append(rank_map_2[model])

    # If there are no common models, you can decide what to return.
    # Here, we'll return None to indicate there's no overlap.
    if not common_models:
        return None, None

    # Compute Spearman’s rank correlation
    correlation, pvalue = spearmanr(ranks_1, ranks_2)

    return correlation, pvalue


if __name__ == "__main__":
    # Example usage with two JSON files on disk:
    with open("Math_3R_5J_10/folder_elo_scores_standalone.json", "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open("Math_9R_9J_10/folder_elo_scores_standalone.json", "r", encoding="utf-8") as f:
        data2 = json.load(f)

    rho, pval = compute_spearman_from_json(data1, data2)

    if rho is not None:
        print("Spearman correlation:", rho)
        print("p-value:", pval)
    else:
        print("No common models found between the two JSON files.")