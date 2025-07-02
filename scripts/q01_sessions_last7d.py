#!/usr/bin/env python3
"""
q01_sessions_avg_per_week.py
----------------------------
Beantwoordt survey-vraag 1 per donor:

    "How many separate ChatGPT sessions did you have in the last 7 days?"

→ In plaats van alleen de laatste 7 dagen
   berekenen we HET GEMIDDELDE aantal sessies per week
   over de hele dataset (ISO-weken).

Antwoordcategorieën:
    0 / 1-2 / 3-5 / 6-10 / more than 10
-----------------------------------------------------------------------
• Eén “session” = één uniek conversation_id (ChatGPT-gesprek)
"""

import pathlib, datetime
import pandas as pd

PARSED   = pathlib.Path("parsed/all.jsonl")      # parser-output
OUT_XLS = pathlib.Path("results/answers_q01.xlsx")  # centrale resultaten

# ------------------------------------------------------------------ #
# Helper: mapping gemiddelde → survey-categorie
# ------------------------------------------------------------------ #
def to_category(avg: float) -> str:
    if avg == 0:
        return "0"
    elif avg < 3:
        return "1-2"
    elif avg < 6:
        return "3-5"
    elif avg < 11:
        return "6-10"
    else:
        return "more than 10"

# ------------------------------------------------------------------ #
# 1. Lees alle Q+A-regels
# ------------------------------------------------------------------ #
df = pd.read_json(PARSED, lines=True)

# Sessiedatum uit timestamp
df["session_date"] = pd.to_datetime(df["question_time"], unit="s").dt.date

# 2. Bepaal ISO-weekcode (YYYY-WW)
iso = pd.to_datetime(df["session_date"]).dt.isocalendar()
df["year_week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
# voorbeeld: 2025-27

# 3. Unieke sessions per donor per week
weekly = (
    df.groupby(["donor_id", "year_week"])["conversation_id"]
      .nunique()
      .reset_index(name="sessions_in_week")
)

# 4. Gemiddelde sessions per week per donor
per_donor = (
    weekly.groupby("donor_id")["sessions_in_week"]
          .mean()
          .reset_index(name="avg_sessions_per_week")
)

# 5. Zet om naar survey-categorie
per_donor["category"]        = per_donor["avg_sessions_per_week"].apply(to_category)
per_donor["survey_question"] = "q01_sessions_avg_per_week"
per_donor["timestamp"]       = datetime.datetime.utcnow()

print(per_donor.head())

# ------------------------------------------------------------------ #
# 6. Wegschrijven / bijschrijven naar Excel
# ------------------------------------------------------------------ #
OUT_XLS.parent.mkdir(exist_ok=True)
mode = "a" if OUT_XLS.exists() else "w"
writer_kwargs = {"mode": mode, "engine": "openpyxl"}
if mode == "a":
    writer_kwargs["if_sheet_exists"] = "overlay"

with pd.ExcelWriter(OUT_XLS, **writer_kwargs) as xls:
    per_donor.to_excel(xls, index=False, header=(mode == "w"))

print(f"✅ Resultaten toegevoegd aan {OUT_XLS}")

