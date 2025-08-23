#!/usr/bin/env python3
"""
build_summary_multi.py
----------------------
• Verwerkt answers_q06 … answers_q11 (meerdere antwoorden mogelijk)
• Splitst de “category”-kolom op "|" en telt ieder label apart
• Schrijft tabbladen Q06 … Q11 naar results/summary_multiselect.xlsx
"""

import pathlib, pandas as pd

RESULTS_DIR = pathlib.Path("results")
OUT_FILE    = RESULTS_DIR / "summary_multiselect.xlsx"

QUESTION_TEXT = {
    "q06": "What tasks do you usually use ChatGPT?",
    "q07": "…Writing & professional communication – sub-tasks",
    "q08": "…Brainstorming & personal ideas / fun – prompt types",
    "q09": "…Coding / programming help – sub-tasks",
    "q10": "…Language practice or translation – main use",
    "q11": "…Study revision / exam prep – sub-tasks",
}

writer = pd.ExcelWriter(OUT_FILE, engine="openpyxl")

for f in sorted(RESULTS_DIR.glob("answers_q0[6-9].xlsx")) + \
         sorted(RESULTS_DIR.glob("answers_q1[0-1].xlsx")):
    q_id   = f.stem.split("_")[1]        # q06 … q11
    sheet  = q_id.upper()               # tabnaam
    print(f"→ verwerken {f.name}")

    df = pd.read_excel(f)               # kolommen: donor_id | category | …

    # -------- split & explode -----------------------------------------
    exploded = (
        df["category"]
          .str.split("|")               # lijstje
          .explode()
          .str.strip()
          .rename("option")
          .to_frame()
    )

    counts = (exploded["option"]
              .value_counts()
              .sort_index()
              .rename("Count")
              .to_frame())

    counts["Count %"] = (counts["Count"] / counts["Count"].sum() * 100
                         ).round(0).astype(int).astype(str) + "%"

    counts.reset_index(inplace=True)    # kolommen: option | Count | Count %
    counts = counts.rename(columns={"option": QUESTION_TEXT[q_id]})
    counts = counts[[QUESTION_TEXT[q_id], "Count %", "Count"]]

    counts.to_excel(writer, sheet_name=sheet, index=False)

writer.close()
print(f"✅ Multiselect-samenvatting → {OUT_FILE}")

