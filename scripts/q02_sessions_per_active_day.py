#!/usr/bin/env python3
"""
q02_sessions_per_active_day.py
------------------------------
Beantwoordt survey-vraag 2 per donor:

    "On a typical day, how many ChatGPT sessions do you start?"
    Antwoorden: 0 / 1 / 2-3 / 4-5 / 6 or more

Werkwijze:
  • Tel het aantal unieke conversation_id per kalenderdag.
  • Neem het GEMIDDELDE over alle 'actieve dagen' (dagen met ≥1 sessie).
  • Sla gemiddeldes van 0 uit (komt praktisch niet voor) of >0 in categorie.
"""

import pathlib, datetime
import pandas as pd

PARSED  = pathlib.Path("parsed/all.jsonl")
OUT_XLS = pathlib.Path("results/answers.xlsx")

# ------------------------------------------------ helper -------------
def to_category(avg: float) -> str:
    if avg == 0:
        return "0"
    elif avg < 2:          # >=1 en <2
        return "1"
    elif avg < 4:          # 2-3
        return "2-3"
    elif avg < 6:          # 4-5
        return "4-5"
    else:                  # 6+
        return "6 or more"

# ------------------------------------------------ analyse ------------
df = pd.read_json(PARSED, lines=True)

# datumkolom
df["day"] = pd.to_datetime(df["question_time"], unit="s").dt.date

# sessies per donor per dag
daily = (
    df.groupby(["donor_id", "day"])["conversation_id"]
      .nunique()
      .reset_index(name="sessions_in_day")
)

# gemiddelde per actieve dag
per_donor = (
    daily.groupby("donor_id")["sessions_in_day"]
         .mean()
         .reset_index(name="avg_sessions_per_active_day")
)

per_donor["category"] = per_donor["avg_sessions_per_active_day"].apply(to_category)
per_donor["survey_question"] = "q02_sessions_per_active_day"
per_donor["timestamp"] = datetime.datetime.utcnow()

print(per_donor.head())

# ------------------------------------------------ excel append -------
OUT_XLS.parent.mkdir(exist_ok=True)
mode = "a" if OUT_XLS.exists() else "w"
writer_kwargs = {"mode": mode, "engine": "openpyxl"}
if mode == "a":
    writer_kwargs["if_sheet_exists"] = "overlay"

with pd.ExcelWriter(OUT_XLS, **writer_kwargs) as xls:
    per_donor.to_excel(xls, index=False, header=(mode == "w"))

print(f"✅ Resultaten toegevoegd aan {OUT_XLS}")

