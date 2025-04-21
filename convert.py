import json
import os
import uuid
from glob import glob

def convert_benchmark_to_dearena(evaluation_files_pattern, output_dir="judgements_mt_bench"):
    """
    Convert Benchmark evaluation results to De-arena voting records format.
    
    Args:
        evaluation_files_pattern: Glob pattern to match evaluation JSON files
        output_dir: Directory to save the converted voting records
    """
    # Get all evaluation files
    evaluation_files = glob(evaluation_files_pattern)
    
    for file_path in evaluation_files:
        # Load the evaluation results
        with open(file_path, 'r') as f:
            evaluation_data = json.load(f)
        
        # Extract basic information
        model_a_id = evaluation_data["original_debate"]["participants"]["model_a"]
        model_b_id = evaluation_data["original_debate"]["participants"]["model_b"]
        
        # Get judge IDs
        judge_ids = evaluation_data["evaluation"]["judges"]
        
        # Process each round's judgments
        for round_num, round_data in evaluation_data["evaluation"]["results"]["round_judgments"].items():
            # Get question ID from round number
            question_id = int(round_num)
            
            # Process each judge's evaluations
            for judge_id in judge_ids:
                if judge_id not in round_data["judgments"]:
                    continue
                    
                judgment = round_data["judgments"][judge_id]
                
                # Skip entries with parsing errors
                if isinstance(judgment, dict) and "parse_error" in judgment:
                    continue
                
                # Create judge directory if it doesn't exist
                judge_dir = os.path.join(output_dir, judge_id)
                os.makedirs(judge_dir, exist_ok=True)
                
                voting_records_path = os.path.join(judge_dir, "voting_records.jsonl")
                
                # Initialize or load existing records
                if os.path.exists(voting_records_path):
                    with open(voting_records_path, 'r') as f:
                        try:
                            existing_records = json.loads(f.read())
                        except:
                            existing_records = []
                else:
                    existing_records = []
                
                # Get the winner
                if judgment.get("winner") == "A":
                    winner = model_a_id
                elif judgment.get("winner") == "B":
                    winner = model_b_id
                else:
                    winner = "TIE"
                
                # Generate a unique ID for this judgment
                data_id = str(uuid.uuid4())
                
                # Create a record in De-arena format
                record = {
                    "response_A": model_a_id,
                    "response_B": model_b_id,
                    "Won": winner,
                    "question_id": question_id,
                    "data_id": data_id
                }
                
                # Add new judgment to existing records
                existing_records.append(record)
                
                # Save updated records
                with open(voting_records_path, 'w') as f:
                    f.write(json.dumps(existing_records))
                
    print(f"Converted {len(evaluation_files)} Benchmark evaluation files to De-arena format")

# Example usage
convert_benchmark_to_dearena("evaluation_results_*.json")