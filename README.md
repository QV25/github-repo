# Thesis Project (repro pack)

## Quickstart
conda env create -f environment.yml
conda activate thesis-parser

## Run order
python scripts/chatgpt_parser.py "data/raw/*.json" --out parsed/all.jsonl
python scripts/q01_sessions_last7d.py
python scripts/q02_sessions_per_active_day.py
python scripts/q03_avg_daily_minutes.py
python scripts/q04_most_common_time.py
python scripts/q05_prompt_length.py
python scripts/q06_embed.py
python scripts/q07_embed_write_subtasks.py
python scripts/q08_embed_brainstorm_subtasks.py
python scripts/q09_embed_code_subtasks.py
python scripts/q10_embed_language_subtasks.py
python scripts/q11_embed_study_subtasks.py
python scripts/q_export_part2_part3.py

## Notes
- Raw exports in data/raw/ (not tracked)
- Set OPENAI_API_KEY via .env (not tracked)
