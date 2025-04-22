from model_handler import get_chat_model
from model_interactions import ModelParticipant, Evaluation, JudgmentCriteria, FinalAssessment
import json
import datetime

def judge_orchestration(json_file_path, judges, project_name):
    # 1. Load the debate results from the JSON file
    with open(json_file_path, 'r') as f:
        debate_data = json.load(f)
    
    # 2. Extract the necessary information
    topic = debate_data["topic"]
    model_a_id = debate_data["participants"]["model_a"]
    model_b_id = debate_data["participants"]["model_b"]
    transcript = debate_data["results"]["transcript"]
    
    # 3. Create judge participants
    judges = [
        ModelParticipant(judge, role="judge") for judge in judges
    ]
    
    # 4. Initialize the evaluation
    evaluation = Evaluation(
        judges=judges,
        transcript=transcript,
        model_a_id=model_a_id,
        model_b_id=model_b_id,
        criteria=["question quality", "answer accuracy", "reasoning depth", "history usage"]
    )
    
    # 5. Run the evaluation
    results = evaluation.run_battle()
    
    # 6. Save the evaluation results
    filename = f"{project_name}/judgements/evaluation_results_{model_a_id}_{model_b_id}.json"
    
    # Create a complete results dictionary
    full_results = {
        "original_debate": {
            "topic": topic,
            "participants": {
                "model_a": model_a_id,
                "model_b": model_b_id
            }
        },
        "evaluation": {
            "judges": [judge.model_id for judge in judges],
            "timestamp": datetime.datetime.now().isoformat(),
            "results": results
        }
    }
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nEvaluation results saved to {filename}")
    
    # 7. Print a summary of the evaluation
    print(f"\n=== Evaluation Summary for Debate on: {topic} ===\n")
    print(f"Model A: {model_a_id}")
    print(f"Model B: {model_b_id}")
    print(f"Overall Winner: {results.get('overall_winner', 'Unknown')}")
    print(f"Round Wins - Model A: {results.get('battle_summary', {}).get('model_a_wins', 0)}")
    print(f"Round Wins - Model B: {results.get('battle_summary', {}).get('model_b_wins', 0)}")
    print(f"Ties: {results.get('battle_summary', {}).get('ties', 0)}")
    print(f"Better History User: {results.get('better_history_user', 'Unknown')}")

if __name__ == "__main__":
    judge_debate_results("debate_results_20250416_153630.json")