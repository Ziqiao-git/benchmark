import os
import json
import argparse
from Spearman_correlation import compute_spearman_correlation
import pandas as pd

def find_judge_ranking_files(folder_path):
    """
    Find all judge ranking files in a folder.
    Returns a dictionary mapping judge names to file paths.
    """
    judge_files = {}
    
    if not os.path.isdir(folder_path):
        return judge_files
    
    for filename in os.listdir(folder_path):
        if filename.startswith("folder_elo_scores_") and filename.endswith("_only_standalone.json"):
            # Extract judge name from filename
            # Format: folder_elo_scores_{judge_name}_only_standalone.json
            judge_name = filename[len("folder_elo_scores_"):-len("_only_standalone.json")]
            file_path = os.path.join(folder_path, filename)
            judge_files[judge_name] = file_path
    
    return judge_files

def compare_judge_correlations(folder1_path, folder2_path):
    """
    Compare Spearman correlations for all judges between two folders.
    Returns a list of results with correlation data.
    """
    # Find judge files in both folders
    folder1_judges = find_judge_ranking_files(folder1_path)
    folder2_judges = find_judge_ranking_files(folder2_path)
    
    # Find common judges
    common_judges = set(folder1_judges.keys()) & set(folder2_judges.keys())
    
    results = []
    
    print(f"Found {len(folder1_judges)} judge files in {os.path.basename(folder1_path)}")
    print(f"Found {len(folder2_judges)} judge files in {os.path.basename(folder2_path)}")
    print(f"Comparing {len(common_judges)} common judges...\n")
    
    for judge_name in sorted(common_judges):
        file1 = folder1_judges[judge_name]
        file2 = folder2_judges[judge_name]
        
        try:
            correlation, p_value = compute_spearman_correlation(file1, file2)
            
            if correlation is not None:
                # Determine correlation strength
                abs_corr = abs(correlation)
                if abs_corr > 0.9:
                    strength = "Very Strong"
                elif abs_corr > 0.7:
                    strength = "Strong"
                elif abs_corr > 0.5:
                    strength = "Moderate"
                elif abs_corr > 0.3:
                    strength = "Weak"
                else:
                    strength = "Very Weak"
                
                direction = "Positive" if correlation > 0 else "Negative"
                interpretation = f"{strength} {direction}"
                
                results.append({
                    'judge': judge_name,
                    'correlation': float(correlation),
                    'p_value': float(p_value),
                    'interpretation': interpretation,
                    'abs_correlation': float(abs_corr),
                    'significant': bool(p_value < 0.05 if p_value is not None else False)
                })
                
                print(f"✅ {judge_name:35s} | {correlation:6.3f} | {p_value:8.2e} | {interpretation}")
            else:
                results.append({
                    'judge': judge_name,
                    'correlation': None,
                    'p_value': None,
                    'interpretation': 'Error',
                    'abs_correlation': 0.0,
                    'significant': False
                })
                print(f"❌ {judge_name:35s} | ERROR  | N/A      | Could not compute")
                
        except Exception as e:
            results.append({
                'judge': judge_name,
                'correlation': None,
                'p_value': None,
                'interpretation': f'Error: {str(e)[:20]}...',
                'abs_correlation': 0.0,
                'significant': False
            })
            print(f"❌ {judge_name:35s} | ERROR  | N/A      | {str(e)[:30]}...")
    
    return results

def display_summary_statistics(results):
    """Display summary statistics for all correlations."""
    valid_results = [r for r in results if r['correlation'] is not None]
    
    if not valid_results:
        print("\n❌ No valid correlations computed.")
        return
    
    correlations = [r['correlation'] for r in valid_results]
    abs_correlations = [r['abs_correlation'] for r in valid_results]
    significant_count = sum(1 for r in valid_results if r['significant'])
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total judges compared: {len(results)}")
    print(f"Valid correlations: {len(valid_results)}")
    print(f"Significant correlations (p < 0.05): {significant_count}")
    print(f"")
    print(f"Correlation Statistics:")
    print(f"  Mean correlation: {sum(correlations)/len(correlations):6.3f}")
    print(f"  Mean |correlation|: {sum(abs_correlations)/len(abs_correlations):6.3f}")
    print(f"  Min correlation: {min(correlations):6.3f}")
    print(f"  Max correlation: {max(correlations):6.3f}")
    
    # Count by strength
    strengths = {}
    for r in valid_results:
        strength = r['interpretation'].split()[0] + " " + r['interpretation'].split()[1]
        strengths[strength] = strengths.get(strength, 0) + 1
    
    print(f"\nCorrelation Strength Distribution:")
    for strength, count in sorted(strengths.items()):
        print(f"  {strength:15s}: {count:2d} judges")

def display_top_bottom_judges(results, n=5):
    """Display top and bottom judges by correlation strength."""
    valid_results = [r for r in results if r['correlation'] is not None]
    valid_results.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"TOP {n} MOST CONSISTENT JUDGES (Highest |correlation|)")
    print(f"{'='*80}")
    
    for i, result in enumerate(valid_results[:n], 1):
        print(f"{i:2d}. {result['judge']:35s} | {result['correlation']:6.3f} | {result['interpretation']}")
    
    print(f"\n{'='*80}")
    print(f"TOP {n} LEAST CONSISTENT JUDGES (Lowest |correlation|)")
    print(f"{'='*80}")
    
    for i, result in enumerate(valid_results[-n:], 1):
        print(f"{i:2d}. {result['judge']:35s} | {result['correlation']:6.3f} | {result['interpretation']}")

def save_results_to_files(results, folder1_name, folder2_name, output_dir="."):
    """Save results to JSON and CSV files."""
    
    # Prepare data for saving
    output_data = {
        "comparison": f"{folder1_name} vs {folder2_name}",
        "total_judges": len(results),
        "valid_correlations": len([r for r in results if r['correlation'] is not None]),
        "results": results
    }
    
    # Save JSON
    json_filename = f"judge_correlations_{folder1_name}_vs_{folder2_name}.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Save CSV for easy analysis
    csv_filename = f"judge_correlations_{folder1_name}_vs_{folder2_name}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Convert to DataFrame
    df_data = []
    for r in results:
        df_data.append({
            'judge': r['judge'],
            'correlation': r['correlation'],
            'p_value': r['p_value'],
            'abs_correlation': r['abs_correlation'],
            'interpretation': r['interpretation'],
            'significant': r['significant']
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_path, index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   📄 {json_path}")
    print(f"   📊 {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare Spearman correlations for all judges between two test folders")
    parser.add_argument("folder1", help="First test folder (e.g., test_results_14)")
    parser.add_argument("folder2", help="Second test folder (e.g., test_results_15)")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top/bottom judges to display")
    parser.add_argument("--output-dir", default=".", help="Directory to save output files")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")
    args = parser.parse_args()
    
    folder1_path = args.folder1
    folder2_path = args.folder2
    
    # Validate folders exist
    if not os.path.isdir(folder1_path):
        print(f"❌ Error: Folder '{folder1_path}' does not exist.")
        return
    
    if not os.path.isdir(folder2_path):
        print(f"❌ Error: Folder '{folder2_path}' does not exist.")
        return
    
    folder1_name = os.path.basename(folder1_path)
    folder2_name = os.path.basename(folder2_path)
    
    print("🔍 JUDGE CORRELATION COMPARISON")
    print("="*80)
    print(f"Comparing judge rankings between:")
    print(f"  📁 {folder1_path}")
    print(f"  📁 {folder2_path}")
    print("="*80)
    print(f"{'Judge Name':35s} | {'Corr':6s} | {'P-value':8s} | {'Interpretation'}")
    print("-"*80)
    
    # Compare correlations
    results = compare_judge_correlations(folder1_path, folder2_path)
    
    # Display analysis
    display_summary_statistics(results)
    display_top_bottom_judges(results, args.top_n)
    
    # Save results
    if not args.no_save:
        try:
            save_results_to_files(results, folder1_name, folder2_name, args.output_dir)
        except ImportError:
            print("⚠️  Warning: pandas not available. Saving JSON only.")
            # Save just JSON
            output_data = {
                "comparison": f"{folder1_name} vs {folder2_name}",
                "total_judges": len(results),
                "valid_correlations": len([r for r in results if r['correlation'] is not None]),
                "results": results
            }
            json_filename = f"judge_correlations_{folder1_name}_vs_{folder2_name}.json"
            json_path = os.path.join(args.output_dir, json_filename)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {json_path}")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main() 