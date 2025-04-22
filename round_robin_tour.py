import itertools
import json
import uuid
import os
from tqdm import tqdm
from model_interactions import ModelParticipant, Debate
from convert import convert_benchmark_to_dearena
from Debate_orchestration import debate_orchestration
from Judge_orchestration import judge_orchestration
from glob import glob
def round_robin_tour(debate_models, judge_models, topic, rounds=2, project_name = "Temp_project"):
    # Create a list of all possible pairings of models
    pairings = list(itertools.combinations(debate_models, 2))
    os.makedirs(project_name, exist_ok=True)
    # Run debates for each pairing
    os.makedirs(f"{project_name}/debates", exist_ok=True)
    for pairing in tqdm(pairings):
        debate_orchestration(pairing, topic, rounds, project_name)
    # Run judge for each debate
    os.makedirs(f"{project_name}/judgements", exist_ok=True)
    for file in glob(f"{project_name}/debates/*.json"):
        judge_orchestration(file, judge_models, project_name)
    # Run automatic arena
    for file in glob(f"{project_name}/judgements/*.json"):
        convert_benchmark_to_dearena(file, output_dir=f"{project_name}/voting_records")

if __name__ == "__main__":
    debate_models = ["gpt4o", "claude"]
    judge_models = ["gpt4o", "claude"]
    round_robin_tour(debate_models, judge_models, topic = "AI and the future of work", rounds=2, project_name="ai_and_work")