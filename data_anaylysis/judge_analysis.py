import os
import json
import argparse
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def analyze_judge_votes(folder_path):
    """
    Analyze debate files to extract judge voting patterns across all rounds.
    
    Args:
        folder_path: Path to folder containing debate JSON files
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")
    
    # Store results
    results = []
    
    # Process each debate file
    for file_path in folder.glob("debate_*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "error" in data:
                print(f"Skipping {file_path} (has error)")
                continue
                
            # Get the debate participants from the file name
            file_name = file_path.name
            parts = file_name.replace("debate_", "").replace(".json", "").split("_vs_")
            if len(parts) != 2:
                print(f"Skipping {file_path} (cannot determine participants)")
                continue
                
            model_a, model_b = parts[0], parts[1]
            
            # Extract round judgments
            round_judgments = data.get("round_judgments", {})
            if not round_judgments:
                print(f"No round_judgments found in {file_path}")
                continue
            
            # Process each round
            for round_num, round_info in round_judgments.items():
                # Check for judgments first (individual judge votes)
                judgments = round_info.get("judgments", {})
                if judgments:
                    for judge, judgment in judgments.items():
                        # Extract the winner from the judgment (could be in different formats)
                        vote = None
                        
                        # Try different possible structures
                        if isinstance(judgment, str):
                            vote = judgment
                        elif isinstance(judgment, dict):
                            # Check various possible keys
                            for key in ["winner", "vote", "preference"]:
                                if key in judgment:
                                    vote = judgment[key]
                                    break
                        
                        # Map "A"/"B" votes to actual model names
                        if vote == "A":
                            vote = model_a
                        elif vote == "B":
                            vote = model_b
                        elif vote == "tie" or vote == "Tie":
                            vote = "tie"
                            
                        results.append({
                            "debate": f"{model_a} vs {model_b}",
                            "model_a": model_a,
                            "model_b": model_b,
                            "round": round_num,
                            "judge": judge,
                            "vote": vote
                        })
                    continue
                
                # If no judgments, try other keys
                # Try vote counts by model (elo_ranking.py style)
                votes = round_info.get("votes", {})
                if votes and len(votes) == 2 and all(k in [model_a, model_b] for k in votes.keys()):
                    # This is aggregate vote data, not per-judge
                    print(f"Found aggregate votes in round {round_num} of {file_path}, but no per-judge data")
                    continue
                
                # Try judge_votes if available
                judge_votes = round_info.get("judge_votes", {})
                if judge_votes:
                    for judge, vote in judge_votes.items():
                        # Map "A"/"B" votes to actual model names if needed
                        if vote == "A":
                            vote = model_a
                        elif vote == "B":
                            vote = model_b
                        elif vote == "tie" or vote == "Tie":
                            vote = "tie"
                            
                        results.append({
                            "debate": f"{model_a} vs {model_b}",
                            "model_a": model_a,
                            "model_b": model_b,
                            "round": round_num,
                            "judge": judge,
                            "vote": vote
                        })
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return results


def create_visualizations(df, judge_model_pivot, output_prefix=None):
    """Create visualizations from the judge voting data"""
    
    # Set the style for all plots
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))
    
    # 1. Judge vote distribution - bar chart
    ax = judge_model_pivot.drop(['Total'] + [c for c in judge_model_pivot.columns if '%' in c], axis=1).plot(
        kind='bar', 
        stacked=True,
        colormap='viridis',
        title='Vote Distribution by Judge'
    )
    ax.set_xlabel('Judge')
    ax.set_ylabel('Number of Votes')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_prefix:
        plt.savefig(f"{output_prefix}_judge_vote_distribution.png")
    plt.show()
    
    # 2. Judge vote percentages - heatmap
    percentage_cols = [c for c in judge_model_pivot.columns if '%' in c and c != 'Total %']
    if percentage_cols:
        plt.figure(figsize=(12, 8))
        percentage_df = judge_model_pivot[percentage_cols].copy()
        percentage_df.columns = [c.replace(' %', '') for c in percentage_df.columns]
        
        # Convert from decimals to percentages
        percentage_df = percentage_df * 100
        
        ax = sns.heatmap(
            percentage_df, 
            annot=True, 
            fmt='.1f',
            cmap='RdYlGn',
            linewidths=.5,
            cbar_kws={'label': 'Vote Percentage (%)'}
        )
        ax.set_title('Judge Voting Preference Heatmap (%)')
        plt.tight_layout()
        
        if output_prefix:
            plt.savefig(f"{output_prefix}_judge_vote_heatmap.png")
        plt.show()
    
    # 3. Overall vote distribution - pie chart
    model_votes = df['vote'].value_counts()
    if 'tie' in model_votes:
        model_votes = model_votes.drop('tie')  # Exclude ties for clearer comparison
    
    plt.figure(figsize=(10, 8))
    ax = model_votes.plot(
        kind='pie',
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        explode=[0.05] * len(model_votes),
        title='Overall Vote Distribution Across All Judges',
    )
    ax.set_ylabel('')
    plt.tight_layout()
    
    if output_prefix:
        plt.savefig(f"{output_prefix}_overall_vote_distribution.png")
    plt.show()
    
    # 4. Model win rate by judge - horizontal bar chart
    judge_model_matrix = df.pivot_table(
        index='judge',
        columns='vote',
        aggfunc='size',
        fill_value=0
    )
    
    # Normalize to get percentages
    judge_model_pct = judge_model_matrix.div(judge_model_matrix.sum(axis=1), axis=0) * 100
    
    # Drop the 'tie' column if it exists
    if 'tie' in judge_model_pct.columns:
        judge_model_pct = judge_model_pct.drop('tie', axis=1)
        
    plt.figure(figsize=(12, 10))
    ax = judge_model_pct.plot(
        kind='barh',
        stacked=True,
        colormap='tab20',
        title='Model Win Rate by Judge'
    )
    ax.set_xlabel('Percentage')
    ax.set_ylabel('Judge')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add percentage labels
    for i, (idx, row) in enumerate(judge_model_pct.iterrows()):
        x_pos = 0
        for col, val in row.items():
            if val > 5:  # Only add text for segments wider than 5%
                ax.text(x_pos + val/2, i, f'{val:.1f}%', 
                        ha='center', va='center', color='black', fontweight='bold')
            x_pos += val
    
    plt.tight_layout()
    
    if output_prefix:
        plt.savefig(f"{output_prefix}_model_win_rate_by_judge.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze judge voting patterns in debate files")
    parser.add_argument("folder", help="Path to folder containing debate JSON files")
    parser.add_argument("--output", help="Path to save the results CSV (optional)")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    args = parser.parse_args()
    
    print(f"Analyzing debates in folder: {args.folder}")
    results = analyze_judge_votes(args.folder)
    
    if not results:
        print("No judge voting data found in the debates.")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Print summary statistics
    print(f"\nFound {len(df)} judge votes across {df['debate'].nunique()} debates and {df['round'].nunique()} rounds")
    
    # JUDGE VOTING PATTERNS SUMMARY
    print("\n==== JUDGE VOTING PATTERNS SUMMARY ====")
    
    # For each judge, count votes for each model
    judge_vote_counts = defaultdict(lambda: defaultdict(Counter))
    
    for _, row in df.iterrows():
        judge = row['judge']
        debate = row['debate']
        model_a = row['model_a']
        model_b = row['model_b']
        vote = row['vote']
        
        if vote == model_a:
            judge_vote_counts[judge][debate][model_a] += 1
        elif vote == model_b:
            judge_vote_counts[judge][debate][model_b] += 1
        else:
            # tie or other vote
            judge_vote_counts[judge][debate]['tie'] += 1
    
    # Print summary for each judge
    for judge, debate_counts in judge_vote_counts.items():
        print(f"\nJudge: {judge}")
        
        for debate, model_counts in debate_counts.items():
            models = debate.split(" vs ")
            model_a, model_b = models[0], models[1]
            
            a_votes = model_counts[model_a]
            b_votes = model_counts[model_b]
            tie_votes = model_counts['tie']
            
            total_votes = a_votes + b_votes + tie_votes
            
            print(f"  Debate: {debate}")
            print(f"    {model_a}: {a_votes} votes ({a_votes/total_votes:.1%})")
            print(f"    {model_b}: {b_votes} votes ({b_votes/total_votes:.1%})")
            if tie_votes > 0:
                print(f"    Tie: {tie_votes} votes ({tie_votes/total_votes:.1%})")
    
    # Create a pivot table: judges x models showing vote counts
    print("\n==== OVERALL JUDGE PREFERENCES ====")
    # Reshape data for pivot table
    vote_records = []
    for judge, debate_counts in judge_vote_counts.items():
        for debate, model_counts in debate_counts.items():
            for model, count in model_counts.items():
                if model != 'tie':  # exclude ties from this summary
                    vote_records.append({
                        'judge': judge,
                        'model': model,
                        'votes': count
                    })
    
    if vote_records:
        votes_df = pd.DataFrame(vote_records)
        judge_model_pivot = votes_df.pivot_table(
            index='judge', 
            columns='model', 
            values='votes', 
            aggfunc='sum',
            fill_value=0
        )
        
        # Add row totals and percentages
        judge_model_pivot['Total'] = judge_model_pivot.sum(axis=1)
        
        # Create percentage columns for each model
        for model in judge_model_pivot.columns:
            if model != 'Total':
                judge_model_pivot[f"{model} %"] = judge_model_pivot[model] / judge_model_pivot['Total']
        
        print(judge_model_pivot)
        
        # Create visualizations
        if not args.no_plots:
            output_prefix = args.output.replace('.csv', '') if args.output else None
            create_visualizations(df, judge_model_pivot, output_prefix)
    
    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        
        # Also save the summary table
        if vote_records:
            summary_output = args.output.replace('.csv', '_summary.csv')
            judge_model_pivot.to_csv(summary_output)
            print(f"\nResults saved to {args.output} and summary to {summary_output}")
        else:
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()