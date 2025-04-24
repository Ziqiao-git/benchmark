from Debate_orchestration import debate_orchestration

from model_interactions import ModelParticipant, Debate

# Creating model participants with custom parameters
model_b = ModelParticipant("local_fallback", 
                          role="debater", 
                          model_path="/lustre/scratch/users/guangyi.liu/ziqiao/models/Qwen2.5-32B-Instruct", 
                          model_name="Qwen2.5-32B-Instruct", 
                          custom_model_id="Qwen2.5-32B-Instruct")

model_a = ModelParticipant("local_fallback", 
                          role="debater", 
                          model_path="/lustre/scratch/users/guangyi.liu/ziqiao/models/QwQ-32B", 
                          model_name="QwQ-32B", 
                          custom_model_id="QwQ-32B")




topic = "AI and the future of work"

debate_orchestration([model_a, model_b], topic, rounds=2, project_name = topic)