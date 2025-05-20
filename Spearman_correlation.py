import json
import argparse
from scipy.stats import spearmanr

def compute_spearman_correlation(file_path1, file_path2):
    """
    Calculate Spearman correlation between rankings in two JSON files.
    Each JSON file should contain a 'ranking' list with objects 
    that have 'rank' and 'model' fields.
    """
    # Load the first JSON file
    with open(file_path1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    
    # Load the second JSON file
    with open(file_path2, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    
    # Extract {model -> rank} mappings from each JSON
    ranking_list1 = data1.get("ranking", [])
    ranking_list2 = data2.get("ranking", [])
    
    rank_map1 = {entry["model"]: entry["rank"] for entry in ranking_list1}
    rank_map2 = {entry["model"]: entry["rank"] for entry in ranking_list2}
    
    # Get sets of models from each file
    models1 = set(rank_map1.keys())
    models2 = set(rank_map2.keys())
    
    # Check if models are the same across both files
    if models1 != models2:
        print("Error: The models in the two JSON files are not identical")
        print(f"Models only in first file: {models1 - models2}")
        print(f"Models only in second file: {models2 - models1}")
        return None, None
    
    # Find the common models (should be all models if the above check passes)
    common_models = models1 & models2
    
    if len(common_models) < 2:
        print("Error: Not enough overlapping models to compute Spearman correlation.")
        return None, None
    
    # Build parallel arrays of ranks for correlation
    ranks1 = []
    ranks2 = []
    for model in common_models:
        ranks1.append(rank_map1[model])
        ranks2.append(rank_map2[model])
    
    # Compute Spearman correlation
    correlation, p_value = spearmanr(ranks1, ranks2)
    
    return correlation, p_value

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Calculate Spearman correlation between model rankings in two JSON files."
    )
    parser.add_argument("file1", help="Path to first JSON file with model rankings")
    parser.add_argument("file2", help="Path to second JSON file with model rankings")
    args = parser.parse_args()
    
    # Compute correlation between the two files
    correlation, p_value = compute_spearman_correlation(args.file1, args.file2)
    
    # Print results
    if correlation is not None:
        print(f"\nComparing rankings between:")
        print(f"  - {args.file1}")
        print(f"  - {args.file2}")
        print(f"\nSpearman correlation: {correlation:.3f}")
        print(f"P-value: {p_value:.3g}")
        
        # Interpret the correlation strength
        if abs(correlation) > 0.9:
            strength = "Very strong"
        elif abs(correlation) > 0.7:
            strength = "Strong"
        elif abs(correlation) > 0.5:
            strength = "Moderate"
        elif abs(correlation) > 0.3:
            strength = "Weak"
        else:
            strength = "Very weak"
            
        print(f"Interpretation: {strength} {'positive' if correlation > 0 else 'negative'} correlation")

if __name__ == "__main__":
    main()
