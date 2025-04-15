# orchestration.py
from model_interactions import ModelParticipant, Debate, Evaluation
from model_handler import get_chat_model
def run_model_debate():
    # Create participants
    bot_a = ModelParticipant("deepseek", role="debater")
    bot_b = ModelParticipant("gpt4o", role="debater")
    judge = ModelParticipant("claude", role="judge")
    
    # Set up a debate
    topic = "Frontiers in Physics: The implications of quantum gravity for our understanding of spacetime"
    debate = Debate(topic, [bot_a, bot_b], judge, rounds=3)
    
    # Run debate
    results = debate.run()
    
    # Print results
    print("=== Debate Transcript ===")
    for entry in results["transcript"]:
        print(f"Round {entry['round']} - {entry['participant']}:")
        print(entry['response'])
        print()
    
    print("=== Judgment ===")
    print(results["judgment"])

def run_model_evaluation():
    # Create judges
    judges = [
        ModelParticipant("grok", role="judge"),
        ModelParticipant("gemini", role="judge"),
        ModelParticipant("claude", role="judge")
    ]
    
    # Set up evaluation
    evaluation = Evaluation(judges)
    
    # Get responses from models
    prompt = "Explain the concept of quantum entanglement to a high school student."
    response_a = get_chat_model("deepseek").generate_messages([("user", prompt)])
    response_b = get_chat_model("gpt4o").generate_messages([("user", prompt)])
    
    # Evaluate responses
    results = evaluation.evaluate_pair(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        model_a_id="deepseek",
        model_b_id="gpt4o"
    )
    
    # Print results
    print("=== Evaluation Results ===")
    print(f"Winner: {results['winner']}")
    print(f"Vote counts: {results['vote_counts']}")
    print("\n=== Judge Explanations ===")
    for judge_id, explanation in results["explanations"].items():
        print(f"{judge_id}:")
        print(explanation)
        print()

if __name__ == "__main__":
    run_model_debate()
    # run_model_evaluation()