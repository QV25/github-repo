#!/usr/bin/env python3
"""
q04_most_common_time.py
-----------------------
Survey-vraag 4 per donor:

    "When do you most often use ChatGPT?"
    Antwoorden:
        • During work - study hours  (Mo–Fr 09:00–18:00)
        • Evenings                   (18:00–03:00)
        • Weekends                   (Sat–Sun all day)
        • Anytime throughout the day (geen >50% dominante categorie)

Werkwijze:
1. Label elke Q+A-regel met time_bucket.
2. Tel per donor de verdeling.
3. Kies dominante bucket  ≥50%; anders 'Anytime ...'.
"""

import pathlib, datetime, pandas as pd

PARSED  = pathlib.Path("parsed/all.jsonl")
OUT_XLS = pathlib.Path("results/answers_q04.xlsx")

# ------------------------------------------------ tijd → bucket -------

def bucket(ts: pd.Timestamp) -> str:
    wd = ts.weekday()          # 0=Mon … 6=Sun
    hour = ts.hour
    if wd >= 5:                # weekend
        return "Weekends"
    if 9 <= hour < 18:
        return "During work - study hours"
    # avond/nacht (18–24 + 0–3)
    if hour >= 18 or hour < 3:
        return "Evenings"
    # overige uren vallen weinig voor (03–09); kwalificeren ook als work/study
    return "During work - study hours"

# ------------------------------------------------ analyse -------------

df = pd.read_json(PARSED, lines=True)
df["ts"] = pd.to_datetime(df["question_time"], unit="s")
df["bucket"] = df["ts"].apply(bucket)

# 1. counts per donor per bucket
counts = (
    df.groupby(["donor_id", "bucket"])
      .size()
      .unstack(fill_value=0)
)

# 2. bepaal dominante bucket of 'Anytime'
def dominant(row):
    total = row.sum()
    top_bucket = row.idxmax()
    if row[top_bucket] / total >= 0.5:
        return top_bucket
    return "Anytime throughout the day"

result = counts.apply(dominant, axis=1).reset_index(name="category")
result["survey_question"] = "q04_most_common_time"
result["timestamp"] = datetime.datetime.utcnow()

print(result.head())

# ------------------------------------------------ wegschrijven --------
OUT_XLS.parent.mkdir(exist_ok=True)
result.to_excel(OUT_XLS, index=False)
print(f"✅ Resultaten opgeslagen in {OUT_XLS}")

