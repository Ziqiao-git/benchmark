from model_handler import get_chat_model
from model_interactions import ModelParticipant, Debate
import json
import datetime

def simple_debate_example():
    # 1. Create two model participants for the debate
    model_a = ModelParticipant("gpt4o", role="debater")
    model_b = ModelParticipant("claude", role="debater")
    
    # 2. Define the debate topic
    topic = "The future of renewable energy"
    
    # 3. Set up the debate with 2 rounds
    debate = Debate(topic, [model_a, model_b], rounds=2)
    
    # 4. Run the debate
    results = debate.run()
    
    # 5. Print a summary of the debate
    print(f"\n=== Debate on: {topic} ===\n")
    print(f"Number of rounds: {debate.rounds}")
    print(f"Number of entries: {len(results['transcript'])}")
    
    # 6. Save the results to a JSON file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debate_results_{timestamp}.json"
    
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
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    simple_debate_example()