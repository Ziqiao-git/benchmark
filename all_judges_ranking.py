import os
import json
import argparse
from judge_only_ranking import (
    MODELS, 
    extract_judge_votes, 
    process_judge_votes, 
    process_debate_file_judge_only,
    write_elo_file
)

# List of all available judges (using the same list as MODELS since any model can be a judge)
ALL_JUDGES = [
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
    # 7. Qwen2.5-72B-Instruct
    "openrouter-qwen-2-72b-instruct",
    # 8. llama-3.3-70b-instruct
    "openrouter-meta-llama-llama-3.3-70b-instruct",
    # 9. Claude-3.5
    "openrouter-claude-3.5-haiku",
    # 10. mistralai/mixtral-8x7b-instruct
    "openrouter-mistral-8x7b-instruct",
    # 11. Gemma-2-27B
    "openrouter-google-gemma-2-27b-it",
    # 12. Mistral-7b-instructv02
    "openrouter-mistralai-mistral-7b-instruct-v0.2",
    # 13. Gemma-2-9B
    "openrouter-google-gemma-2-9b-it",
    # 14. microsoft/phi-4-reasoning-plus
    "openrouter-phi-4-reasoning-plus",
    # 15. QwQ-32B
    "openrouter-QwQ-32B",
]

def run_judge_ranking(judge_name, folders, base_dir, mode="standalone"):
    """
    Run ELO ranking for a specific judge across given folders.
    Returns the final ELO ratings for that judge.
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING JUDGE: {judge_name.upper()}")
    print(f"{'='*60}")
    
    if mode == "standalone":
        # Fresh ELO for each folder
        all_judge_elos = {}
        
        for folder_name in folders:
            folder_path = os.path.join(base_dir, folder_name)
            if not os.path.isdir(folder_path):
                print(f"  Folder not found: {folder_path}, skipping...")
                continue
                
            print(f"  Processing folder: {folder_name}")
            
            # Reset fresh ELO for each folder
            judge_elo = {m: 1200 for m in MODELS}
            
            # Load failed pairs
            failed_pairs = load_failed_pairs(folder_path)
            
            # Process debates
            debate_files = [f for f in os.listdir(folder_path) if f.endswith(".json") and f.startswith("debate_")]
            processed_count = 0
            
            for fname in debate_files:
                debate_path = os.path.join(folder_path, fname)
                old_elo = judge_elo.copy()
                process_debate_file_judge_only(debate_path, failed_pairs, judge_elo, judge_name)
                
                # Check if any ELO changed
                if judge_elo != old_elo:
                    processed_count += 1
            
            print(f"    Processed {processed_count} debates with {judge_name} judgments")
            
            # Store this folder's results
            all_judge_elos[folder_name] = judge_elo.copy()
            
            # Write individual folder result
            output_path = os.path.join(folder_path, f"folder_elo_scores_{judge_name}_only_standalone.json")
            write_elo_file(judge_elo, output_path, judge_name)
        
        return all_judge_elos
    
    else:  # aggregator mode
        # Accumulate ELO across all folders
        judge_elo = {m: 1200 for m in MODELS}
        
        for folder_name in folders:
            folder_path = os.path.join(base_dir, folder_name)
            if not os.path.isdir(folder_path):
                print(f"  Folder not found: {folder_path}, skipping...")
                continue
                
            print(f"  Processing folder: {folder_name}")
            
            # Load failed pairs
            failed_pairs = load_failed_pairs(folder_path)
            
            # Process debates
            debate_files = [f for f in os.listdir(folder_path) if f.endswith(".json") and f.startswith("debate_")]
            processed_count = 0
            
            for fname in debate_files:
                debate_path = os.path.join(folder_path, fname)
                old_elo = judge_elo.copy()
                process_debate_file_judge_only(debate_path, failed_pairs, judge_elo, judge_name)
                
                # Check if any ELO changed
                if judge_elo != old_elo:
                    processed_count += 1
            
            print(f"    Processed {processed_count} debates with {judge_name} judgments")
        
        return {f"aggregated_across_all": judge_elo}

def load_failed_pairs(folder_path):
    """Load failed pairs from state.json if it exists."""
    failed_pairs_path = os.path.join(folder_path, "state.json")
    failed_pairs = []
    if os.path.isfile(failed_pairs_path):
        try:
            with open(failed_pairs_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)
                failed_pairs = state_data.get("failed_pairs", [])
        except Exception as e:
            print(f"    Warning: Could not load state.json: {e}")
    return failed_pairs

def display_judge_summary(all_judges_results, top_n=5):
    """Display a summary of all judges' rankings."""
    print(f"\n{'='*80}")
    print(f"SUMMARY: TOP {top_n} MODELS BY EACH JUDGE")
    print(f"{'='*80}")
    
    for judge_name, judge_results in all_judges_results.items():
        print(f"\n🔹 JUDGE: {judge_name}")
        
        for folder_name, elo_ratings in judge_results.items():
            print(f"  📁 {folder_name}:")
            
            # Sort models by ELO
            sorted_models = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
            
            for i, (model, elo) in enumerate(sorted_models[:top_n], 1):
                print(f"    {i}. {model}: {elo:.1f}")

def save_comprehensive_report(all_judges_results, output_file="all_judges_ranking_report.json"):
    """Save a comprehensive report of all judges' rankings."""
    report = {
        "description": "ELO rankings for all judges across all folders",
        "judges": all_judges_results,
        "summary": {}
    }
    
    # Create summary statistics
    for judge_name, judge_results in all_judges_results.items():
        report["summary"][judge_name] = {}
        
        for folder_name, elo_ratings in judge_results.items():
            sorted_models = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
            report["summary"][judge_name][folder_name] = {
                "top_3": [{"model": m, "elo": round(e, 2)} for m, e in sorted_models[:3]],
                "bottom_3": [{"model": m, "elo": round(e, 2)} for m, e in sorted_models[-3:]],
                "elo_range": round(sorted_models[0][1] - sorted_models[-1][1], 2)
            }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Comprehensive report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run ELO rankings for all judges individually")
    parser.add_argument("--judges", nargs="+", help="Specific judges to process (default: all)", default=ALL_JUDGES)
    parser.add_argument("--folders", nargs="+", help="Folders to process")
    parser.add_argument("--base-dir", help="Base directory (default: current working directory)")
    parser.add_argument("--mode", choices=["standalone", "aggregator"], default="standalone",
                       help="Standalone: fresh ELO per folder, Aggregator: accumulate across folders")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top models to show in summary")
    parser.add_argument("--output", help="Output file for comprehensive report", default="all_judges_ranking_report.json")
    args = parser.parse_args()

    base_dir = args.base_dir or os.getcwd()
    
    # Determine which folders to process
    if args.folders:
        folders = args.folders
    else:
        # Auto-detect test_results_* folders
        folders = [d for d in os.listdir(base_dir) 
                  if d.startswith("test_results_") and os.path.isdir(os.path.join(base_dir, d))]
        if not folders:
            print("No test_results_* folders found. Please specify --folders explicitly.")
            return

    print("🚀 ALL JUDGES RANKING CALCULATOR")
    print(f"📁 Processing folders: {', '.join(folders)}")
    print(f"👨‍⚖️ Processing judges: {', '.join(args.judges)}")
    print(f"⚙️ Mode: {args.mode}")
    
    # Run rankings for all judges
    all_judges_results = {}
    
    for judge_name in args.judges:
        try:
            judge_results = run_judge_ranking(judge_name, folders, base_dir, args.mode)
            all_judges_results[judge_name] = judge_results
        except Exception as e:
            print(f"❌ Error processing judge {judge_name}: {e}")
            continue
    
    # Display summary
    display_judge_summary(all_judges_results, args.top_n)
    
    # Save comprehensive report
    save_comprehensive_report(all_judges_results, args.output)
    
    print(f"\n✅ Completed processing {len(all_judges_results)} judges!")

if __name__ == "__main__":
    main() 