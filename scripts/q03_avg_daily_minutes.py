#!/usr/bin/env python3
"""
q03_avg_daily_minutes.py
------------------------
Survey-vraag 3 per donor:

    "On average, how long does a single ChatGPT session last?"
    Antwoord­categorieën:
        <5 min / 5-15 / 15-30 / 30-60 / >60 min

Stappen:
1. Sessieduur per conversation_id (min. 3 min).
2. Som sessieduren per donor per dag.
3. Gemiddelde (min) over alle actieve dagen.
4. Omzetten naar survey-categorie.
"""

import pathlib, datetime
import pandas as pd

PARSED  = pathlib.Path("parsed/all.jsonl")
OUT_XLS = pathlib.Path("results/answers_q03.xlsx")

# ------------------------------------------------ helper -------------
def to_category(mins: float) -> str:
    if mins < 5:
        return "Less than 5 minutes"
    elif mins < 15:
        return "5-15 minutes"
    elif mins < 30:
        return "15-30 minutes"
    elif mins < 60:
        return "30-60 minutes"
    else:
        return "More than 60 minutes"

MIN_SEC = 180      # min. 3 minuten

# ------------------------------------------------ analyse ------------
df = pd.read_json(PARSED, lines=True)

# sorteer per donor + conversation + turn_index (zekerheid)
df.sort_values(["donor_id", "conversation_id", "turn_index"], inplace=True)

# 1. tel Q+A-paren (regels) per conversation
conv_sizes = (
    df.groupby(["donor_id", "conversation_id"])
      .size()
      .reset_index(name="num_pairs")
)

# 2. sessieduur = 3 min per paar
conv_sizes["session_minutes"] = conv_sizes["num_pairs"] * 3

# 3. plak startdatum (eerste vraag) erbij
df["ts"] = df["question_time"].fillna(df["answer_time"])

first_dates = (
    df.groupby(["donor_id", "conversation_id"])["ts"]
      .min()
      .pipe(pd.to_datetime, unit="s")   # → datetime64
      .dt.date                          # → pure date
      .reset_index(name="date")
)
conv = conv_sizes.merge(first_dates, on=["donor_id", "conversation_id"])

# 4. totale minuten per donor per dag
daily = (
    conv.groupby(["donor_id", "date"])["session_minutes"]
        .sum()
        .reset_index(name="minutes_in_day")
)

# 5. gemiddelde minuten per actieve dag
per_donor = (
    daily.groupby("donor_id")["minutes_in_day"]
         .mean()
         .reset_index(name="avg_minutes_per_active_day")
)

# 6. categorie
per_donor["category"]        = per_donor["avg_minutes_per_active_day"].apply(to_category)
per_donor["survey_question"] = "q03_avg_daily_minutes"
per_donor["timestamp"]       = datetime.datetime.utcnow()

print(per_donor.head())

# ------------------------------------------------ wegschrijven -------
OUT_XLS.parent.mkdir(exist_ok=True)
per_donor.to_excel(OUT_XLS, index=False)
print(f"✅ Resultaten opgeslagen in {OUT_XLS}")

