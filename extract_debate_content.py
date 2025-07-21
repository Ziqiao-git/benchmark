#!/usr/bin/env python3
"""
Extract debate content from test_results_* folders and save to debates subfolders.

This script processes all test_results_* folders, extracts the debate transcript 
content from each debate JSON file, and saves it to a "debates" subfolder 
within each test_results_* folder.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def extract_debate_transcript(debate_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only the debate transcript content from a debate JSON file.
    
    Args:
        debate_data: The full debate data dictionary
        
    Returns:
        Dictionary containing only the debate transcript and metadata
    """
    # Extract basic info
    participants = set()
    specific_topic = None
    general_topic = None
    detailed_instructions = None
    
    # Get participants from transcript
    if "transcript" in debate_data:
        for entry in debate_data["transcript"]:
            if "participant" in entry:
                participants.add(entry["participant"])
        
        # Extract the specific topic/question from the first challenger's response
        for entry in debate_data["transcript"]:
            if entry.get("role") == "challenger" and entry.get("round") == 1 and entry.get("step") == 1:
                specific_topic = entry.get("response", "")
                break
    
    # Get general topic and detailed instructions from top level if available
    if "topic" in debate_data:
        general_topic = debate_data["topic"]
    
    if "detailed_instructions" in debate_data:
        detailed_instructions = debate_data["detailed_instructions"]
    
    # Create clean debate content
    debate_content = {
        "general_topic": general_topic,
        "detailed_instructions": detailed_instructions,
        "specific_topic": specific_topic,
        "participants": sorted(list(participants)),
        "transcript": debate_data.get("transcript", [])
    }
    
    return debate_content


def process_debate_file(debate_file_path: Path, output_dir: Path) -> bool:
    """
    Process a single debate file and save the transcript to the output directory.
    
    Args:
        debate_file_path: Path to the debate JSON file
        output_dir: Directory to save the extracted debate content
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the debate data
        with open(debate_file_path, 'r', encoding='utf-8') as f:
            debate_data = json.load(f)
        
        # Extract transcript content
        debate_content = extract_debate_transcript(debate_data)
        
        # Create output filename
        output_filename = f"debate_transcript_{debate_file_path.stem}.json"
        output_path = output_dir / output_filename
        
        # Save the extracted content
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(debate_content, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"Error processing {debate_file_path}: {e}")
        return False


def process_test_results_folder(test_results_dir: Path) -> None:
    """
    Process a single test_results_* folder and extract all debate content.
    
    Args:
        test_results_dir: Path to the test_results_* folder
    """
    print(f"Processing folder: {test_results_dir.name}")
    
    # Create debates output directory
    debates_dir = test_results_dir / "debates"
    debates_dir.mkdir(exist_ok=True)
    
    # Find all debate JSON files
    debate_files = []
    for pattern in ["debate_*.json", "debate_results_*.json"]:
        debate_files.extend(list(test_results_dir.glob(pattern)))
    
    if not debate_files:
        print(f"  No debate files found in {test_results_dir.name}")
        return
    
    # Process each debate file
    successful = 0
    failed = 0
    
    for debate_file in debate_files:
        if process_debate_file(debate_file, debates_dir):
            successful += 1
        else:
            failed += 1
    
    print(f"  Processed {successful} files successfully, {failed} failed")
    
    # Create summary file
    summary_path = debates_dir / "extraction_summary.json"
    summary_data = {
        "source_folder": test_results_dir.name,
        "total_files_processed": len(debate_files),
        "successful_extractions": successful,
        "failed_extractions": failed,
        "debate_files": [f.name for f in debate_files]
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)


def main():
    """Main function to process all test_results_* folders."""
    parser = argparse.ArgumentParser(
        description="Extract debate content from test_results_* folders"
    )
    parser.add_argument(
        "--root-dir", 
        default=".", 
        help="Root directory to search for test_results_* folders (default: current directory)"
    )
    parser.add_argument(
        "--folder-pattern",
        default="test_results_*",
        help="Pattern to match test result folders (default: test_results_*)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path object
    root_dir = Path(args.root_dir).resolve()
    
    if not root_dir.exists():
        print(f"Error: Root directory {root_dir} does not exist")
        return
    
    # Find all test_results_* folders
    test_results_folders = sorted(list(root_dir.glob(args.folder_pattern)))
    
    if not test_results_folders:
        print(f"No folders matching pattern '{args.folder_pattern}' found in {root_dir}")
        return
    
    print(f"Found {len(test_results_folders)} folders matching pattern '{args.folder_pattern}'")
    print(f"Root directory: {root_dir}")
    print()
    
    # Process each folder
    for folder in test_results_folders:
        if folder.is_dir():
            process_test_results_folder(folder)
        else:
            print(f"Skipping non-directory: {folder}")
    
    print("\n✅ Debate content extraction completed!")
    print(f"Debate transcripts saved to 'debates' subfolders within each test_results_* folder")


if __name__ == "__main__":
    main() 