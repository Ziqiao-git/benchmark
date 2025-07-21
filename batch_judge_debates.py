#!/usr/bin/env python3
"""
Judge existing AIME debates using existing infrastructure
"""
import asyncio
import os
import json
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from model_interactions import ModelParticipant
from async_orchestration import AsyncDebate_and_Judge

# Reuse judges list from your existing files
judges = [
    "deepseek", "o1", "openrouter-Qwen3-235B-A22B", "openrouter-claude-3.7-sonnet-thinking",
    "gpt4o", "openrouter-deepseek-v3-0324", "openrouter-qwen-2-72b-instruct",
    "openrouter-meta-llama-llama-3.3-70b-instruct", "openrouter-claude-3.5-haiku",
    "openrouter-mistral-8x7b-instruct", "openrouter-google-gemma-2-27b-it",
    "openrouter-mistralai-mistral-7b-instruct-v0.2", "openrouter-google-gemma-2-9b-it",
    "openrouter-phi-4-reasoning-plus", "openrouter-QwQ-32B"
]

async def judge_single_debate(debate_file_path: str, judge_participants: list):
    """Judge a single debate file using existing AsyncDebate_and_Judge.judge_transcript_only()"""
    try:
        print(f"  📄 Judging: {os.path.basename(debate_file_path)}")
        
        # Load debate data to get participants info
        with open(debate_file_path, 'r', encoding='utf-8') as f:
            debate_data = json.load(f)
        
        participants = debate_data.get("participants", [])
        if len(participants) != 2:
            return {"error": "invalid_participants", "file": debate_file_path}
        
        model_a_id, model_b_id = participants
        transcript = debate_data.get("transcript", [])
        rounds = max([entry.get("round", 1) for entry in transcript])
        
        # Create mock participants for AsyncDebate_and_Judge
        mock_participants = [
            ModelParticipant(model_a_id, role="participant"),
            ModelParticipant(model_b_id, role="participant")
        ]
        
        # Create AsyncDebate_and_Judge instance  
        debate_judge = AsyncDebate_and_Judge(
            participants=mock_participants,
            rounds=rounds,
            transcript=transcript,
            instruction_set=["AIME Problem", "Mathematical competition problem"],
            judges_list=judge_participants,
            auto_judge=False,
            results_dir=os.path.dirname(debate_file_path)
        )
        
        # Use existing judge_transcript_only method!
        result = await debate_judge.judge_transcript_only()
        
        print(f"    ✅ Judged by {len(judge_participants)} judges")
        return {"status": "success", "file": debate_file_path, "result": result}
        
    except Exception as e:
        print(f"    ❌ Error: {str(e)}")
        return {"status": "error", "file": debate_file_path, "error": str(e)}

async def process_folder(folder_path: str, judge_participants: list):
    """Process all debate files in a folder - reusing extract_debate_content.py pattern"""
    print(f"\n📁 Processing: {os.path.basename(folder_path)}")
    
    if not os.path.isdir(folder_path):
        print(f"  ❌ Folder not found: {folder_path}")
        return
    
    # Find debate files using the same pattern as extract_debate_content.py
    debate_files = []
    for pattern in ["debate_*.json", "debate_results_*.json"]:
        debate_files.extend(list(Path(folder_path).glob(pattern)))
    
    if not debate_files:
        print(f"  ⚠️  No debate files found")
        return
    
    print(f"  Found {len(debate_files)} debate files")
    
    # Process files with concurrency control (like async_test_2.py)
    sem = asyncio.Semaphore(3)  # Limit concurrent judgings
    
    async def judge_with_sem(debate_file):
        async with sem:
            return await judge_single_debate(str(debate_file), judge_participants)
    
    # Use tqdm for progress tracking (like async_test_2.py)
    tasks = [asyncio.create_task(judge_with_sem(f)) for f in debate_files]
    results = await tqdm_asyncio.gather(*tasks, desc=f"Judging {os.path.basename(folder_path)}")
    
    successful = len([r for r in results if r.get("status") == "success"])
    print(f"  📊 {successful}/{len(results)} files judged successfully")

async def main():
    """Main function - reusing folder discovery pattern"""
    base_dir = Path(".")
    
    # Auto-detect AIME folders (like extract_debate_content.py)
    aime_folders = []
    for item in base_dir.iterdir():
        if item.is_dir() and ("AIME" in item.name or item.name.startswith("test_results_AIME")):
            aime_folders.append(item)
    
    if not aime_folders:
        print("❌ No AIME folders found")
        return
    
    aime_folders = sorted(aime_folders)
    print(f"🎯 Found {len(aime_folders)} AIME folders: {[f.name for f in aime_folders]}")
    
    # Create judge participants (reusing your existing pattern)
    judge_participants = [ModelParticipant(judge_id, role="judge") for judge_id in judges]
    print(f"👨‍⚖️ Using {len(judge_participants)} judges")
    
    # Process each folder
    for folder in aime_folders:
        await process_folder(str(folder), judge_participants)
    
    print("\n✅ All AIME folders processed!")

if __name__ == "__main__":
    asyncio.run(main()) 