#!/usr/bin/env python3
"""
build_summary_python.py
-----------------------
Bundelt alle `results/answers_q*.xlsx` tot één Excel
`results/summary_python.xlsx`, precies in survey-tabel-opmaak.

• 1 sheet per vraag (tabnaam = Q01, Q02, …)
• Kolommen:  category | Count % | Count
"""

import pathlib, pandas as pd

RESULTS_DIR = pathlib.Path("results")
OUT_FILE    = RESULTS_DIR / "summary_python.xlsx"

# mapping vraag-id → volledige vraagtekst
QUESTION_TEXT = {
    "q01": "How many separate ChatGPT sessions did you have in the last 7 days?",
    "q02": "On a typical day, how many ChatGPT sessions do you start?",
    "q03": "On average, how long does a single ChatGPT session last?",
    "q04": "When do you most often use ChatGPT?",
    "q05": "How long are your typical prompts?",
    "q06": "What tasks do you usually use ChatGPT?",
    "q07": "If you chose “Writing & professional communication”, which sub-tasks do you use ChatGPT for?",
    "q08": "If you chose “Brainstorming & personal ideas / fun”, what kinds of prompts do you ask ChatGPT for?",
    "q09": "If you chose “Coding / programming help”, what coding sub-tasks do you use ChatGPT for?",
    "q10": "If you chose “Language practice or translation”, what do you mainly use ChatGPT for?",
    "q11": "If you chose “Study revision / exam prep”, which study tasks do you use ChatGPT for?",
}

writer = pd.ExcelWriter(OUT_FILE, engine="openpyxl")

for f in sorted(RESULTS_DIR.glob("answers_q*.xlsx")):
    q_id = f.stem.split("_")[1]          # 'q01' t/m 'q11'
    sheet = q_id.upper()                # tabnaam 'Q01'
    print(f"→ verwerken {f.name}")

    df = pd.read_excel(f)               # verwacht kolommen: donor_id | category | …
    # tel
    counts = (
        df["category"]
          .value_counts(dropna=False)
          .sort_index()
          .rename("Count")
          .to_frame()
    )
    counts["Count %"] = (counts["Count"] / counts["Count"].sum() * 100).round(0).astype(int).astype(str) + "%"
    counts = counts[["Count %", "Count"]]            # volgorde gelijk aan screenshot
    counts.index.name = QUESTION_TEXT.get(q_id, "Question")
    counts.reset_index(inplace=True)                 # eerste kolom wordt ‘category’

    # schrijf naar sheet
    counts.to_excel(writer, sheet_name=sheet, index=False)

writer.close()
print(f"✅ Samenvatting klaar → {OUT_FILE}")

