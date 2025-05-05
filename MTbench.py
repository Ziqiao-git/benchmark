from Debate_orchestration import debate_orchestration
from Judge_orchestration import judge_orchestration
from model_interactions import ModelParticipant, Debate
from tqdm import tqdm

topic = "Question that is similar/related to the one in the given instruction "
detailed_instructions = [
["Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.", "Rewrite your previous response. Start every sentence with the letter A."],
["Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. Keep the email short and to the point.", "Take a moment to evaluate and critique your own response."],   
["Imagine you are writing a blog post comparing two popular smartphone models. Develop an outline for the blog post, including key points and subheadings to effectively compare and contrast the features, performance, and user experience of the two models. Please answer in fewer than 200 words.", "Take your previous response and rephrase it as a limerick."],
["Write a persuasive email to convince your introverted friend, who dislikes public speaking, to volunteer as a guest speaker at a local event. Use compelling arguments and address potential objections. Please be concise.", "Can you rephrase your previous answer and incorporate a metaphor or simile in each sentence?"],
["Describe a vivid and unique character, using strong imagery and creative language. Please answer in fewer than two paragraphs.", "Revise your previous response and incorporate an allusion to a famous work of literature or historical event in each sentence."],
["Write a descriptive paragraph about a bustling marketplace, incorporating sensory details such as smells, sounds, and visual elements to create an immersive experience for the reader.", "Rework your previous response. Begin each sentence with the subsequent letter of the alphabet, commencing from B."],
["Could you write a captivating short story beginning with the sentence: The old abandoned house at the end of the street held a secret that no one had ever discovered.", "Now, do the same task again but only use four-word sentences."],
["Craft an intriguing opening paragraph for a fictional short story. The story should involve a character who wakes up one morning to find that they can time travel.", "Summarize the story with three bullet points using only nouns and adjectives, without verbs."],
["Help me construct a catchy, yet scientifically accurate, headline for an article on the latest discovery in renewable bio-energy, while carefully handling the ethical dilemmas surrounding bio-energy sources. Propose 4 options.", "Alter your previous response. Make the following adjustments to the 2nd option: 1. Make the tone sound casual 2. Embed an advertisement for a company called \"FlexPower\" 3. Fewer than 10 words."],
["Edit the following paragraph to correct any grammatical errors:\nShe didn't remembre where is her purse, so I thinks its in the car but he's say it's on kitchen table but he are not sure, and then they asked me to looking for it, she's say, \"Can you?\", and I responds with, \"Maybe, but ain't no sure,\" and he not heard me, and, \"What?\", he asks, \"Did you found it?\".", "Modify your earlier reply and eliminate the use of gendered pronouns."]
]

# Creating model participants with custom parameters
all_models = ["openrouter-claude-3.7-sonnet-thinking", 
              "openrouter-deepseek-v3-0324", 
              "openrouter-Grok-3-Beta", 
              "openrouter-Gemini-2.5-flash-thinking", 
              "openrouter-QwQ-32B", 
              "openrouter-Qwen3-235B-A22B", 
              "openrouter-Gemini-2.5-pro",
              "o1",
              "o3",
              "o4-mini",
              "deepseek",
              "openrouter-Amazon_Nova_1"]

num_instructions = len(detailed_instructions)
num_models = len(all_models)
pairs_count = num_models * (num_models - 1) // 2
total_debates = pairs_count * num_instructions

progress = tqdm(total=total_debates, desc="Debate Progress")

# Loop over each instruction
for instr_idx, instruction in enumerate(detailed_instructions, start=1):
    project_name = f"MTbench_{instr_idx}"

    # Round-robin all model pairs
    for i in range(num_models):
        for j in range(i + 1, num_models):
            model_a_id = all_models[i]
            model_b_id = all_models[j]

            # Create participants
            model_a = ModelParticipant(model_a_id, role="debater")
            model_b = ModelParticipant(model_b_id, role="debater")

            # Single-round debate, using instruction as the "detailed_instructions" param
            debate_orchestration(
                [model_a, model_b],
                topic=topic,
                rounds=1,  # only 1 round
                project_name=project_name,
                detailed_instructions=instruction
            )

            progress.update(1)

progress.close()
print("All debates done! Now you can run the judge step")
    