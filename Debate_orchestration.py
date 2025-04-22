from model_handler import get_chat_model
from model_interactions import ModelParticipant, Debate
import json
import datetime
import os

def debate_orchestration(models, topic, rounds, project_name):
    # 1. Create two model participants for the debate
    model_a = ModelParticipant(models[0], role="debater")
    model_b = ModelParticipant(models[1], role="debater")
    
    # 2. Define the debate topic
    topic = topic
    
    # 3. Set up the debate with 2 rounds
    debate = Debate(topic, [model_a, model_b], rounds=rounds)
    
    # 4. Run the debate
    results = debate.run()
    
    # 5. Print a summary of the debate
    print(f"\n=== Debate on: {topic} ===\n")
    print(f"Number of rounds: {debate.rounds}")
    print(f"Number of entries: {len(results['transcript'])}")
    
    # 6. Save the results to a JSON file
    filename = f"{project_name}/debates/debate_results_{model_a.model_id}_{model_b.model_id}.json"
    
    # Create a more complete results dictionary
    full_results = {
        "topic": topic,
        "participants": {
            "model_a": model_a.model_id,
            "model_b": model_b.model_id
        },
        "rounds": debate.rounds,
        "timestamp": datetime.datetime.now().isoformat(),
        "results": results
    }
    

    # Define filename only once
    filename = f"{project_name}/debates/debate_results_{model_a.model_id}_{model_b.model_id}.json"
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    # here is a simple test
    debate_orchestration(["gpt4o", "claude"], topic = "AI and the future of work", rounds = 2, project_name = "ai_and_work")