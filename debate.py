from Debate_orchestration import debate_orchestration
from Judge_orchestration import judge_orchestration
from model_interactions import ModelParticipant, Debate


# Creating model participants with custom parameters
model_a = ModelParticipant("o3", role="debater")
model_b = ModelParticipant("o4-mini", role="debater")


topic = "Consoling a friend"
detailed_instructions = "The Questioner will take on the role of a Visitor, seeking advice. The Rival will act as a Therapist, offering guidance to the Visitor. The Visitor poses questions or concerns, and the Therapist responds with supportive advice. The primary objective is to evaluate how effectively the participant can provide comfort and reassurance to the Visitor."
debate_orchestration([model_a, model_b], topic, rounds=20, project_name = topic, detailed_instructions = detailed_instructions)
judges = []
judges.append(ModelParticipant("o1", role="judge"))
judges.append(ModelParticipant("gpt4o", role="judge"))
# judges.append(ModelParticipant("claude", role="judge"))
judges.append(ModelParticipant("o3", role="judge"))
judges.append(ModelParticipant("o4-mini", role="judge"))

judge_orchestration(f"{topic}/debates/debate_results_{model_a.model_id}_{model_b.model_id}.json", judges, topic)















# from openai import OpenAI
# client = OpenAI()

# response = client.responses.create(
#   model="o1-pro-2025-03-19",
#   input="hello,introduce your self, what ai model are you in openai?",
#   text={
#     "format": {
#       "type": "text"
#     }
#   },
#   reasoning={
#     "effort": "medium"
#   },
#   tools=[],
#   store=True
# )
# print(response)