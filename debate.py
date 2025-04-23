from Debate_orchestration import debate_orchestration

from model_interactions import ModelParticipant, Debate

# Creating model participants with custom parameters
model_a = ModelParticipant("local_fallback", 
                          role="debater", 
                          model_path="/lustre/scratch/users/guangyi.liu/ziqiao/models/QwQ-32B", 
                          model_name="QwQ-32B", 
                          custom_model_id="QwQ-32B")

model_b = ModelParticipant("local_fallback", 
                          role="debater", 
                          model_path="/lustre/scratch/users/guangyi.liu/ziqiao/models/Athene-V2-Chat", 
                          model_name="aya-expanse-32b", 
                          custom_model_id="aya-expanse-32b")


topic = "AI and the future of work"

debate_orchestration([model_a, model_b], topic, rounds=2, project_name = topic)