#!/usr/bin/env python3
"""
q05_prompt_length.py
--------------------
Survey-vraag 5 per donor:

    "How long are your typical prompts?"

Buckets:
    • Sentence      (≤ 20 words)
    • Paragraph     (21-60 words)
    • Multi-para    (>60 words)
    • Varies        (geen bucket ≥50%)

Werkwijze:
1. Woordcount per Q-prompt.
2. Label in bucket.
3. Tel verdeling per donor.
4. Dominante bucket ≥50% → categorie, anders 'Varies too much to say'.
"""

import pathlib, datetime, re
import pandas as pd

PARSED  = pathlib.Path("parsed/all.jsonl")
OUT_XLS = pathlib.Path("results/answers_q05.xlsx")
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

# ---------- helper: to bucket ----------------------------------------
def bucket(n_words: int) -> str:
    if n_words <= 20:
        return "One short sentence (≤ 20 words)"
    elif n_words <= 60:
        return "A short paragraph (21 – 60 words)"
    else:
        return "Multiple paragraphs (> 60 words)"

# ---------- 1. lees alle Q+A-regels ----------------------------------
df = pd.read_json(PARSED, lines=True)

# woordcount van de prompt-tekst
df["word_count"] = df["question"].apply(lambda txt: len(WORD_RE.findall(txt)))

# bucket per regel
df["bucket"] = df["word_count"].apply(bucket)

# ---------- 2. verdeling per donor -----------------------------------
counts = (
    df.groupby(["donor_id", "bucket"])
      .size()
      .unstack(fill_value=0)
)

# ---------- 3. dominante bucket of 'Varies' --------------------------
def choose(row):
    total = row.sum()
    top = row.idxmax()
    if row[top] / total >= 0.5:
        return top
    return "Varies too much to say"

result = counts.apply(choose, axis=1).reset_index(name="category")
result["survey_question"] = "q05_prompt_length"
result["timestamp"] = datetime.datetime.utcnow()

print(result.head())

# ---------- 4. opslaan -----------------------------------------------
OUT_XLS.parent.mkdir(exist_ok=True)
result.to_excel(OUT_XLS, index=False)
print(f"✅ Resultaten opgeslagen in {OUT_XLS}")

