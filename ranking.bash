# export OPENAI_API="" 
export OVERALL_IDS="81-160"
export SAVE_OUTPUT_FILE_PATH="arabic_mt_ranking_result.txt"
export JUDGE_OPEN_MODEL="gpt4o,claude"

export JUDGE_API_MODEL="gpt4o,claude"
export BASE_MODEL_LIST="gpt4o,claude"


export SORT_MODEL_LIST="gpt4o,claude"

python automatic_arena.py --base_dir "ai_and_work/voting_records"