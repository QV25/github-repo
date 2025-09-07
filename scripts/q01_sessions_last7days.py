#!/usr/bin/env python3
"""
q01_sessions_last7days.py
-------------------------
Beantwoordt survey-vraag Q01:
  "How many separate ChatGPT sessions did you have in the last 7 days?"

Definities:
- Sessie = donor-brede activiteitsepisode. Een nieuwe sessie start
  wanneer de tijd sinds de vorige Q+A voor deze donor > GAP_MIN minuten is.
  (Dus: dezelfde conversation_id op een later moment => nieuwe sessie.)
- Venster = de laatste 7 *dagen* t.o.v. de laatste activiteit in de logs
  van die donor.
- Tijdzone voor daggrenzen: Europe/Amsterdam (voor interpretatie/consistente vensters).
- Resultaat per donor in categorieën: 0, 1-2, 3-5, 6-10, More than 10.

Uitvoer: results/answers_q01_last7days.xlsx
"""

import pathlib
import pandas as pd

PARSED = pathlib.Path("parsed/all.jsonl")
OUT_XLS = pathlib.Path("results/answers_q01_last7days.xlsx")
TZ = "Europe/Amsterdam"
GAP_MIN = 30  # sessie-split drempel (pas aan als gewenst)

def to_category(n: int) -> str:
    if n == 0: return "0"
    if n <= 2: return "1-2"
    if n <= 5: return "3-5"
    if n <= 10: return "6-10"
    return "More than 10"

# 1) Lees Q+A en maak lokale tijden
df = pd.read_json(PARSED, lines=True)
# parser heeft records zonder timestamps al gedropt; we gebruiken vraag-tijd
dt_local = pd.to_datetime(df["question_time"], unit="s", utc=True).dt.tz_convert(TZ)
df = df.assign(dt_local=dt_local).sort_values(["donor_id", "dt_local"]).reset_index(drop=True)

# 2) Bouw sessies donor-breed met inactiviteitsdrempel (ongeacht conversation_id)
df["prev_dt"] = df.groupby("donor_id")["dt_local"].shift(1)
gap = (df["dt_local"] - df["prev_dt"]).dt.total_seconds() / 60.0
df["new_session"] = (df["prev_dt"].isna()) | (gap > GAP_MIN)
# session_id = cumsum over booleans per donor
df["session_id"] = df.groupby("donor_id")["new_session"].cumsum()

# 3) Sessietabel (start/einde per sessie)
sessions = (
    df.groupby(["donor_id", "session_id"], as_index=False)
      .agg(start_local=("dt_local","min"), end_local=("dt_local","max"))
)

# 4) Venster 'laatste 7 dagen' per donor
#    anchor_end = laatste activiteit (einde van de laatste sessie)
anchor = sessions.groupby("donor_id")["end_local"].max().rename("anchor_end")
sessions = sessions.merge(anchor, on="donor_id", how="left")
sessions["window_start"] = sessions["anchor_end"] - pd.Timedelta(days=7)

# Tel alleen sessies die in het venster zijn *gestart*
mask = (sessions["start_local"] >= sessions["window_start"]) & (sessions["start_local"] <= sessions["anchor_end"])
last7_counts = sessions[mask].groupby("donor_id").size().rename("sessions_last7").reset_index()

# Donors zonder sessies in venster krijgen 0
all_donors = sessions[["donor_id"]].drop_duplicates()
last7 = all_donors.merge(last7_counts, on="donor_id", how="left").fillna({"sessions_last7": 0})
last7["sessions_last7"] = last7["sessions_last7"].astype(int)

# 5) Categorieën + metadata
last7["category"] = last7["sessions_last7"].apply(to_category)
last7["survey_question"] = "q01_sessions_last7days"
last7["gap_minutes"] = GAP_MIN
# optioneel: timestamps als strings (Excel-friendly)
last7 = last7.merge(anchor.reset_index(), on="donor_id", how="left")
last7["anchor_end_str"] = last7["anchor_end"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
last7 = last7.drop(columns=["anchor_end"])  # geen tz-aware kolommen naar Excel

# 6) Schrijven
OUT_XLS.parent.mkdir(exist_ok=True, parents=True)
last7.to_excel(OUT_XLS, index=False)
print(f"✅ Q01 (last 7 days) geschreven → {OUT_XLS}  | donors={len(last7)}")

