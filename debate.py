from Debate_orchestration import debate_orchestration
from Judge_orchestration import judge_orchestration
from model_interactions import ModelParticipant, Debate


# Creating model participants with custom parameters
model_a = ModelParticipant("o3", role="debater")
model_b = ModelParticipant("o1", role="debater")


topic = "High level chemistry questions"
detailed_instructions = "Focus on the organic chemistry"
debate_orchestration([model_a, model_b], topic, rounds=3, project_name = topic, detailed_instructions = detailed_instructions)
judges = []
judges.append(ModelParticipant("o1", role="judge"))
judges.append(ModelParticipant("gpt4o", role="judge"))
judges.append(ModelParticipant("claude", role="judge"))
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