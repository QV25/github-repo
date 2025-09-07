#!/usr/bin/env python3
"""
q04_most_common_time.py
-----------------------
Q10: "When do you most often use ChatGPT?"

• Tellen op sessie-niveau (gap-split > GAP_MIN minuten), niet per Q+A
• Lokale tijd: Europe/Amsterdam
• Buckets op sessie-START:
    - Work/study: Mon–Fri 09:00–18:00
    - Evenings:   18:00–03:00 (dagoverschrijdend, elke dag)
    - Other:      alles daarbuiten (diagnostisch, geen eindcategorie)
• Eindcategorie per donor:
    - Als share(work) >= DOMINANCE -> "During work / study hours"
    - else als share(evenings) >= DOMINANCE -> "Evenings"
    - anders -> "Anytime throughout the day"
"""

import pathlib
import pandas as pd

PARSED  = pathlib.Path("parsed/all.jsonl")
OUT_XLS = pathlib.Path("results/answers_q04.xlsx")

TZ = "Europe/Amsterdam"
GAP_MIN = 30       # min zonder activiteit -> nieuwe sessie
DOMINANCE = 0.33   # 50% meerderheid

WORK_START = 9     # 09:00
WORK_END   = 18    # tot 18:00
EVENING_END = 3    # tot 03:00

def build_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Bouw sessies per donor met gap-split en haal start_local op."""
    df = df.sort_values(["donor_id","dt_local"]).copy()
    df["prev_dt"] = df.groupby("donor_id")["dt_local"].shift(1)
    gap = (df["dt_local"] - df["prev_dt"]).dt.total_seconds() / 60.0
    df["new_session"] = df["prev_dt"].isna() | (gap > GAP_MIN)
    df["session_id"] = df.groupby("donor_id")["new_session"].cumsum()
    sessions = (
        df.groupby(["donor_id","session_id"], as_index=False)
          .agg(start_local=("dt_local","min"))
    )
    return sessions

def classify_bucket(ts) -> str:
    wd = ts.weekday()  # 0=Mon..6=Sun
    h  = ts.hour
    # Work/study = Mon-Fri 09:00-18:00
    if wd < 5 and (WORK_START <= h < WORK_END):
        return "work"
    # Evenings = 18:00-03:00 (dagoverschrijdend)
    if (h >= WORK_END) or (h < EVENING_END):
        return "evening"
    # Overig
    return "other"

# 1) Data + lokale tijd (gebruik question_time)
df = pd.read_json(PARSED, lines=True)
df = df.assign(dt_local=pd.to_datetime(df["question_time"], unit="s", utc=True).dt.tz_convert(TZ))

# 2) Sessies bouwen en starttijd classificeren
sessions = build_sessions(df)
sessions["bucket"] = sessions["start_local"].apply(classify_bucket)

# 3) Counts & shares per donor
counts = sessions.pivot_table(index="donor_id", columns="bucket",
                              values="session_id", aggfunc="count", fill_value=0)

# Zorg dat alle kolommen bestaan
for c in ["work","evening","other"]:
    if c not in counts.columns:
        counts[c] = 0

counts = counts.reset_index()
counts["n_sessions"]    = counts[["work","evening","other"]].sum(axis=1)
den = counts["n_sessions"].where(counts["n_sessions"]>0, other=1)  # guard /0
counts["share_work"]    = counts["work"]    / den
counts["share_evening"] = counts["evening"] / den
counts["share_other"]   = counts["other"]   / den

def decide(row):
    if row["n_sessions"] == 0:
        return "Anytime throughout the day"
    if row["share_work"] >= DOMINANCE:
        return "During work / study hours"
    if row["share_evening"] >= DOMINANCE:
        return "Evenings"
    return "Anytime throughout the day"

counts["category"] = counts.apply(decide, axis=1)

# 4) Metadata + schrijven (alleen strings/nums naar Excel)
counts["survey_question"]     = "q04_most_common_time"
counts["gap_minutes"]         = GAP_MIN
counts["dominance_threshold"] = DOMINANCE
counts["timezone"]            = TZ
counts["work_hours"]          = f"Mon-Fri {WORK_START:02d}:00-{WORK_END:02d}:00"
counts["evenings_hours"]      = "Daily 18:00-03:00"

out = counts[[
    "donor_id","n_sessions","work","evening","other",
    "share_work","share_evening","share_other","category",
    "survey_question","gap_minutes","dominance_threshold","timezone","work_hours","evenings_hours"
]]

OUT_XLS.parent.mkdir(exist_ok=True, parents=True)
mode = "a" if OUT_XLS.exists() else "w"
writer_kwargs = {"mode": mode, "engine": "openpyxl"}
if mode == "a":
    writer_kwargs["if_sheet_exists"] = "overlay"
with pd.ExcelWriter(OUT_XLS, **writer_kwargs) as xls:
    out.to_excel(xls, index=False, header=(mode == "w"))

print(f"✅ Q10 geschreven → {OUT_XLS} (n_donors={len(out)})")
