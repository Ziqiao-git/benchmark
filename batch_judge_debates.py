#!/usr/bin/env python3
"""
Batch Judge Debates
==================

A general-purpose script to run judges on existing debate files across multiple folders.
Works with any debate files in the standard format (debate_*.json, debate_results_*.json).

Usage:
    python batch_judge_debates.py                          # Auto-detect all debate folders
    python batch_judge_debates.py --folders test_results_* # Specific folders
    python batch_judge_debates.py --judges deepseek o1     # Specific judges only
    python batch_judge_debates.py --pattern "*AIME*"       # Custom folder pattern
"""
import asyncio
import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm.asyncio import tqdm_asyncio
from model_interactions import ModelParticipant
from async_orchestration import AsyncDebate_and_Judge

# Default judges list (can be overridden via command line)
DEFAULT_JUDGES = [
    "deepseek", "o1", "openrouter-Qwen3-235B-A22B", "openrouter-claude-3.7-sonnet-thinking",
    "gpt4o", "openrouter-deepseek-v3-0324", "openrouter-qwen-2-72b-instruct",
    "openrouter-meta-llama-llama-3.3-70b-instruct", "openrouter-claude-3.5-haiku",
    "openrouter-mistral-8x7b-instruct", "openrouter-google-gemma-2-27b-it",
    "openrouter-mistralai-mistral-7b-instruct-v0.2", "openrouter-google-gemma-2-9b-it",
    "openrouter-phi-4-reasoning-plus", "openrouter-QwQ-32B"
]

async def judge_single_debate(debate_file_path: str, judge_participants: List[ModelParticipant], 
                             instruction_set: Optional[List[str]] = None) -> Dict[str, Any]:
    """Judge a single debate file using existing AsyncDebate_and_Judge.judge_transcript_only()"""
    try:
        # Load debate data to get participants info
        with open(debate_file_path, 'r', encoding='utf-8') as f:
            debate_data = json.load(f)
        
        participants = debate_data.get("participants", {})
        
        # Handle both formats: list ["model1", "model2"] or dict {"model_a": "model1", "model_b": "model2"}
        if isinstance(participants, list):
            if len(participants) != 2:
                return {"error": "invalid_participants", "file": debate_file_path, "details": f"Found {len(participants)} participants"}
            model_a_id, model_b_id = participants
        elif isinstance(participants, dict):
            if "model_a" not in participants or "model_b" not in participants:
                return {"error": "invalid_participants", "file": debate_file_path, "details": f"Missing model_a or model_b in participants"}
            model_a_id = participants["model_a"]
            model_b_id = participants["model_b"]
        else:
            return {"error": "invalid_participants", "file": debate_file_path, "details": f"Participants format not recognized: {type(participants)}"}
        transcript = debate_data.get("transcript", [])
        
        if not transcript:
            return {"error": "no_transcript", "file": debate_file_path}
        
        # Safely determine number of rounds
        try:
            rounds = max([entry.get("round", 1) for entry in transcript if isinstance(entry.get("round"), int)])
        except (ValueError, TypeError):
            rounds = 3  # Default fallback
        
        # Create dummy participants (we don't need actual models for judging)
        # We'll create a simple class that mimics ModelParticipant for AsyncDebate_and_Judge
        class DummyParticipant:
            def __init__(self, model_id, role="participant"):
                self.model_id = model_id
                self.role = role
                self.history = []
        
        mock_participants = [
            DummyParticipant(model_a_id, role="participant"),
            DummyParticipant(model_b_id, role="participant")
        ]
        
        # Use provided instruction_set or infer from data/folder
        if instruction_set is None:
            topic = debate_data.get("topic", "General Debate")
            folder_name = os.path.basename(os.path.dirname(debate_file_path))
            
            if "AIME" in folder_name or "Math" in folder_name:
                instruction_set = [topic, "Mathematical competition problems and reasoning"]
            elif "Code" in folder_name or "CS" in folder_name:
                instruction_set = [topic, "Computer science and programming challenges"]
            elif "Job" in folder_name or "PSYCH" in folder_name:
                instruction_set = [topic, "Professional and psychological assessment"]
            else:
                instruction_set = [topic, "General knowledge and reasoning"]
        
        # Create AsyncDebate_and_Judge instance  
        debate_judge = AsyncDebate_and_Judge(
            participants=mock_participants,
            rounds=rounds,
            transcript=transcript,
            instruction_set=instruction_set,
            judges_list=judge_participants,
            auto_judge=False,
            results_dir=os.path.dirname(debate_file_path)
        )
        
        # Use existing judge_transcript_only method!
        result = await debate_judge.judge_transcript_only()
        
        return {
            "status": "success", 
            "file": debate_file_path, 
            "judges_count": len(judge_participants),
            "rounds": rounds,
            "instruction_set": instruction_set,
            "result": result
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "file": debate_file_path, 
            "error": str(e),
            "error_type": type(e).__name__
        }

async def process_folder(folder_path: str, judge_participants: List[ModelParticipant], 
                        max_concurrent: int = 3) -> Dict[str, Any]:
    """Process all debate files in a folder"""
    folder_name = os.path.basename(folder_path)
    print(f"\n📁 Processing: {folder_name}")
    
    if not os.path.isdir(folder_path):
        error_msg = f"Folder not found: {folder_path}"
        print(f"  ❌ {error_msg}")
        return {"error": "folder_not_found", "path": folder_path, "message": error_msg}
    
    # Find debate files using multiple patterns
    debate_files = []
    patterns = ["debate_*.json", "debate_results_*.json"]
    for pattern in patterns:
        debate_files.extend(list(Path(folder_path).glob(pattern)))
    
    # Remove duplicates and sort
    debate_files = sorted(list(set(debate_files)))
    
    if not debate_files:
        print(f"  ⚠️  No debate files found (searched for: {', '.join(patterns)})")
        return {"warning": "no_debate_files", "path": folder_path, "patterns_searched": patterns}
    
    print(f"  Found {len(debate_files)} debate files")
    
    # Process files with concurrency control
    sem = asyncio.Semaphore(max_concurrent)
    
    async def judge_with_sem(debate_file):
        async with sem:
            return await judge_single_debate(str(debate_file), judge_participants)
    
    # Use tqdm for progress tracking
    start_time = time.time()
    tasks = [asyncio.create_task(judge_with_sem(f)) for f in debate_files]
    results = await tqdm_asyncio.gather(*tasks, desc=f"Judging {folder_name}")
    processing_time = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "error"]
    
    print(f"  📊 {len(successful)}/{len(results)} files judged successfully")
    if failed:
        print(f"  ❌ {len(failed)} files failed")
        # Show first few errors for debugging
        for error_result in failed[:3]:
            print(f"    • {os.path.basename(error_result['file'])}: {error_result.get('error', 'Unknown error')}")
        if len(failed) > 3:
            print(f"    ... and {len(failed) - 3} more")
    
    print(f"  ⏱️  Processing time: {processing_time:.1f}s")
    
    return {
        "folder": folder_path,
        "folder_name": folder_name,
        "total_files": len(debate_files),
        "successful": len(successful),
        "failed": len(failed),
        "processing_time": processing_time,
        "results": results,
        "successful_files": [r["file"] for r in successful],
        "failed_files": [r["file"] for r in failed]
    }

def discover_folders(base_dir: str, pattern: str = None, folders: List[str] = None) -> List[Path]:
    """Discover debate folders based on pattern or explicit list"""
    base_path = Path(base_dir)
    
    if folders:
        # Use explicitly provided folders
        folder_paths = []
        for folder in folders:
            if os.path.isabs(folder):
                folder_paths.append(Path(folder))
            else:
                folder_paths.append(base_path / folder)
        return folder_paths
    
    # Auto-discover based on pattern
    discovered = []
    
    if pattern:
        # Use custom pattern
        import fnmatch
        for item in base_path.iterdir():
            if item.is_dir() and fnmatch.fnmatch(item.name, pattern):
                discovered.append(item)
    else:
        # Default: look for common debate folder patterns
        patterns = ["*AIME*", "test_results_*", "*debate*", "*Code*", "*Math*", "*Job*", "*MT*"]
        for item in base_path.iterdir():
            if item.is_dir():
                for p in patterns:
                    import fnmatch
                    if fnmatch.fnmatch(item.name, p):
                        discovered.append(item)
                        break
    
    return sorted(list(set(discovered)))

async def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Run judges on existing debate files across multiple folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Auto-detect all debate folders
  %(prog)s --folders test_results_AIME*       # Specific folders
  %(prog)s --judges deepseek o1 gpt4o         # Use only specific judges
  %(prog)s --pattern "*AIME*"                 # Custom folder pattern
  %(prog)s --max-concurrent 5                 # Higher concurrency
  %(prog)s --base-dir /path/to/debates        # Different base directory
        """
    )
    
    parser.add_argument("--folders", nargs="+", help="Specific folders to process")
    parser.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES, 
                       help="Specific judges to use (default: all available)")
    parser.add_argument("--pattern", help="Pattern to match folder names (e.g., '*AIME*')")
    parser.add_argument("--base-dir", default=".", help="Base directory to search (default: current directory)")
    parser.add_argument("--max-concurrent", type=int, default=3, 
                       help="Maximum concurrent judgings per folder (default: 3)")
    parser.add_argument("--output", help="Output file for summary report")
    
    args = parser.parse_args()
    
    # Discover folders to process
    folders = discover_folders(args.base_dir, args.pattern, args.folders)
    
    if not folders:
        print("❌ No debate folders found!")
        if args.pattern:
            print(f"   Searched for pattern: {args.pattern}")
        elif args.folders:
            print(f"   Specified folders: {args.folders}")
        else:
            print("   Try specifying --folders or --pattern explicitly")
        return
    
    # Filter out non-existent folders
    existing_folders = [f for f in folders if f.exists() and f.is_dir()]
    if len(existing_folders) != len(folders):
        missing = set(folders) - set(existing_folders)
        print(f"⚠️  Warning: {len(missing)} folders not found: {[str(f) for f in missing]}")
    
    folders = existing_folders
    if not folders:
        print("❌ No valid folders found!")
        return
    
    print(f"🎯 Found {len(folders)} folders: {[f.name for f in folders]}")
    
    # Create judge participants
    judge_participants = [ModelParticipant(judge_id, role="judge") for judge_id in args.judges]
    print(f"👨‍⚖️ Using {len(judge_participants)} judges: {', '.join(args.judges)}")
    print(f"⚙️  Max concurrent per folder: {args.max_concurrent}")
    
    # Process each folder
    start_time = time.time()
    all_results = []
    
    for folder in folders:
        result = await process_folder(str(folder), judge_participants, args.max_concurrent)
        all_results.append(result)
    
    total_time = time.time() - start_time
    
    # Generate summary
    total_files = sum(r.get("total_files", 0) for r in all_results)
    total_successful = sum(r.get("successful", 0) for r in all_results)
    total_failed = sum(r.get("failed", 0) for r in all_results)
    
    print(f"\n🎯 OVERALL SUMMARY:")
    print(f"   Folders processed: {len(folders)}")
    print(f"   Total debate files: {total_files}")
    print(f"   Successfully judged: {total_successful}")
    print(f"   Failed: {total_failed}")
    print(f"   Success rate: {total_successful/total_files*100:.1f}%" if total_files > 0 else "   Success rate: N/A")
    print(f"   Total processing time: {total_time:.1f}s")
    print(f"   Judges used: {len(judge_participants)}")
    
    # Save summary report
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arguments": vars(args),
        "summary": {
            "folders_processed": len(folders),
            "total_files": total_files,
            "successful": total_successful,
            "failed": total_failed,
            "success_rate": total_successful/total_files if total_files > 0 else 0,
            "total_processing_time": total_time,
            "judges_used": args.judges
        },
        "folder_results": all_results
    }
    
    output_file = args.output or f"batch_judge_summary_{int(time.time())}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Summary report saved to: {output_file}")
    print("✅ Batch judging completed!")

if __name__ == "__main__":
    asyncio.run(main()) 