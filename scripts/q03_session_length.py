#!/usr/bin/env python3
"""
q03_session_length.py
---------------------
Q9: "On average, how long does a single ChatGPT session last?"

• Sessie = donor-brede episode; nieuwe sessie bij inactiviteit > GAP_MIN minuten
• Duur per sessie = end_local - start_local (in minuten), geen proxy
• Window: laatste RECENT_DAYS tov donor's laatste activiteit; zo niet, fallback = alle sessies
• Samenvatten per donor met de MEDIANE sessieduur (mean ter context)
• Banding (op de mediaan):
    <5, 5-15, 15-30, 30-60, >60 minuten
"""

import pathlib
import pandas as pd

PARSED   = pathlib.Path("parsed/all.jsonl")
OUT_XLS  = pathlib.Path("results/answers_q03.xlsx")

TZ = "Europe/Amsterdam"
GAP_MIN = 30        # min zonder activiteit -> nieuwe sessie
RECENT_DAYS = 28    # venster voor 'typical' gedrag (zoals Q8)

def to_category(mins: float) -> str:
    if mins < 5:   return "Less than 5 minutes"
    if mins < 15:  return "5-15 minutes"
    if mins < 30:  return "15-30 minutes"
    if mins < 60:  return "30-60 minutes"
    return "More than 60 minutes"

def build_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Bouw sessies per donor (gap-split), en bereken start/end/duration."""
    df = df.sort_values(["donor_id","dt_local"]).copy()
    df["prev_dt"] = df.groupby("donor_id")["dt_local"].shift(1)
    gap = (df["dt_local"] - df["prev_dt"]).dt.total_seconds() / 60.0
    df["new_session"] = df["prev_dt"].isna() | (gap > GAP_MIN)
    df["session_id"] = df.groupby("donor_id")["new_session"].cumsum()

    sessions = (
        df.groupby(["donor_id","session_id"], as_index=False)
          .agg(start_local=("dt_local","min"), end_local=("dt_local","max"))
    )
    # duur in minuten (no negative, just in case)
    sessions["duration_min"] = (sessions["end_local"] - sessions["start_local"]).dt.total_seconds() / 60.0
    sessions["duration_min"] = sessions["duration_min"].clip(lower=0)
    sessions["start_date"] = sessions["start_local"].dt.date
    return sessions

# 1) Data + lokale tijd
df = pd.read_json(PARSED, lines=True)
df = df.assign(dt_local=pd.to_datetime(df["question_time"], unit="s", utc=True).dt.tz_convert(TZ))

# 2) Sessies + anchor per donor
sessions = build_sessions(df)
anchor = sessions.groupby("donor_id")["end_local"].max().rename("anchor_end")
sessions = sessions.merge(anchor, on="donor_id", how="left")

# 3) Window-randen per donor (op datum)
sessions["window_start_date"] = (sessions["anchor_end"] - pd.Timedelta(days=RECENT_DAYS-1)).dt.date
sessions["anchor_end_date"]   = sessions["anchor_end"].dt.date

# 4) Per donor: neem sessies die STARTEN binnen het window; zo niet, fallback = alle sessies
rows = []
for donor, g in sessions.groupby("donor_id"):
    ws = g["window_start_date"].iloc[0]
    we = g["anchor_end_date"].iloc[0]

    in_win = g[(g["start_date"] >= ws) & (g["start_date"] <= we)]
    fallback = False
    use = in_win if not in_win.empty else g.assign() or g
    if in_win.empty:
        fallback = True

    durs = use["duration_min"].astype(float)
    n_used = int(len(durs))
    med = float(durs.median()) if n_used else 0.0
    mean = float(durs.mean()) if n_used else 0.0
    p25 = float(durs.quantile(0.25)) if n_used else 0.0
    p75 = float(durs.quantile(0.75)) if n_used else 0.0

    rows.append({
        "donor_id": donor,
        "sessions_in_window": int(len(in_win)),
        "sessions_used": n_used,
        "median_session_minutes": med,
        "mean_session_minutes": mean,
        "p25_session_minutes": p25,
        "p75_session_minutes": p75,
        "category": to_category(med),
        "fallback_all_history": int(fallback),
    })

out = pd.DataFrame(rows).sort_values("donor_id").reset_index(drop=True)
out["survey_question"] = "q03_session_length"
out["gap_minutes"]     = GAP_MIN
out["recent_days"]     = RECENT_DAYS
out["timestamp_utc"]   = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# 5) Schrijven (append of nieuw) — geen tz-kolommen naar Excel
OUT_XLS.parent.mkdir(exist_ok=True, parents=True)
mode = "a" if OUT_XLS.exists() else "w"
writer_kwargs = {"mode": mode, "engine": "openpyxl"}
if mode == "a":
    writer_kwargs["if_sheet_exists"] = "overlay"
with pd.ExcelWriter(OUT_XLS, **writer_kwargs) as xls:
    out.to_excel(xls, index=False, header=(mode == "w"))

print(f"✅ Q9 geschreven → {OUT_XLS} (n_donors={len(out)})")

