from Debate_orchestration import debate_orchestration
from Judge_orchestration import judge_orchestration
from model_interactions import ModelParticipant, Debate


# Creating model participants with custom parameters
model_a = ModelParticipant("deepseek", role="debater")
model_b = ModelParticipant("claude", role="debater")


topic = "Real Life emergency room doctor"
detailed_instructions = "The questioner will take on the role of a nurse, reporting a patient's condition who is in the emergency room. The rival will act as a doctor, offering guidance to the patient. The nurse just reports the condition, and the doctor responds with decsisions and actions. The primary objective is to evaluate how effectively the participant can provide comfort and reassurance to the patient."
debate_orchestration([model_a, model_b], topic, rounds=1, project_name = topic, detailed_instructions = detailed_instructions)
judges = []
judges.append(ModelParticipant("o1", role="judge"))
judges.append(ModelParticipant("claude", role="judge"))
# judges.append(ModelParticipant("claude", role="judge"))
judges.append(ModelParticipant("o3", role="judge"))
judges.append(ModelParticipant("o4-mini", role="judge"))
judges.append(ModelParticipant("deepseek", role="judge"))

judge_orchestration(f"{topic}/debates/debate_results_{model_a.model_id}_{model_b.model_id}.json", judges, topic, detailed_instructions = detailed_instructions, response_criteria = None, question_criteria = None)



