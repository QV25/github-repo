#!/usr/bin/env python3
"""
q02_sessions_typical_day.py
---------------------------
Q8: "On a typical day, how many ChatGPT sessions do you start?"

• Window: laatste RECENT_DAYS t.o.v. donor's laatste activiteit
• Sessie = donor-brede episode; nieuwe sessie bij inactiviteit > GAP_MIN minuten
• Dag = lokale kalenderdag (Europe/Amsterdam)
• Typical day = mediane sessies per dag over ALLE dagen in het window
  (na verwijderen van ZEER lange 0-streaks: >= ZERO_STREAK_MAX+1)
• Banding (op de mediane waarde):
    <0.5 -> 0
    [0.5,1.5) -> 1
    [1.5,3.5) -> 2-3
    [3.5,5.5) -> 4-5
    ≥5.5 -> 6 or more
"""

import pathlib
import pandas as pd

PARSED  = pathlib.Path("parsed/all.jsonl")
OUT_XLS = pathlib.Path("results/answers_q02.xlsx")

TZ = "Europe/Amsterdam"
GAP_MIN = 30          # minuten inactiviteit -> nieuwe sessie
RECENT_DAYS = 28      # venster voor "typical day"
ZERO_STREAK_MAX = 5  # verwijder 0-streaks STRIKT langer dan dit (>=6)

def to_category(x: float) -> str:
    if x < 0.5:   return "0"
    if x < 1.5:   return "1"
    if x < 3.5:   return "2-3"
    if x < 5.5:   return "4-5"
    return "6 or more"

def build_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Sessies per donor met gap-split."""
    df = df.sort_values(["donor_id","dt_local"]).copy()
    df["prev_dt"] = df.groupby("donor_id")["dt_local"].shift(1)
    gap = (df["dt_local"] - df["prev_dt"]).dt.total_seconds() / 60.0
    df["new_session"] = df["prev_dt"].isna() | (gap > GAP_MIN)
    df["session_id"] = df.groupby("donor_id")["new_session"].cumsum()
    sessions = (
        df.groupby(["donor_id","session_id"], as_index=False)
          .agg(start_local=("dt_local","min"), end_local=("dt_local","max"))
    )
    sessions["start_date"] = sessions["start_local"].dt.date
    return sessions

def drop_long_zero_streaks(s: pd.Series) -> pd.Series:
    """Verwijder aaneengesloten runs van 0 waarvan de lengte > ZERO_STREAK_MAX."""
    is_zero = s.eq(0)
    grp = (is_zero != is_zero.shift(fill_value=False)).cumsum()     # run labels
    run_len = is_zero.groupby(grp).transform("size")
    keep = (~is_zero) | (run_len <= ZERO_STREAK_MAX)
    return s[keep]

# 1) Parse-output + lokale tijd
df = pd.read_json(PARSED, lines=True)
dt_local = pd.to_datetime(df["question_time"], unit="s", utc=True).dt.tz_convert(TZ)
df = df.assign(dt_local=dt_local)

# 2) Sessies bouwen
sessions = build_sessions(df)

# 3) Anchor (laatste activiteit) + window (datums) per donor
anchor = sessions.groupby("donor_id")["end_local"].max().rename("anchor_end")
sessions = sessions.merge(anchor, on="donor_id", how="left")
sessions["window_start_date"] = (sessions["anchor_end"] - pd.Timedelta(days=RECENT_DAYS-1)).dt.date
sessions["anchor_end_date"]   = sessions["anchor_end"].dt.date

# 4) Sessies per dag tellen (start telt in die dag) — ongefilterd, filter pas per donor
daily_all = (
    sessions.groupby(["donor_id","start_date"])
            .size()
            .rename("sessions_in_day")
            .reset_index()
)

# 5) Per donor: dagreeks binnen het WINDOW maken en veilig vullen met counts in dat window
rows = []
for donor, g in sessions.groupby("donor_id"):
    ws = g["window_start_date"].iloc[0]
    we = g["anchor_end_date"].iloc[0]

    # volledige kalenderdagreeks in window
    day_idx = pd.date_range(ws, we, freq="D").date
    s = pd.Series(0, index=day_idx, dtype="int64")

    # telwaarden voor deze donor BINNEN het window
    g_daily = daily_all[daily_all["donor_id"] == donor]
    g_daily_win = g_daily[(g_daily["start_date"] >= ws) & (g_daily["start_date"] <= we)]

    if not g_daily_win.empty:
        # zorg dat de index-typen overeenkomen (datetime.date)
        counts = (g_daily_win
                  .set_index("start_date")["sessions_in_day"]
                  .copy())
        counts.index = pd.Index([pd.to_datetime(d).date() for d in counts.index])
        # veilig toevoegen: reindex + add (voorkomt KeyError)
        s = s.add(counts.reindex(s.index, fill_value=0), fill_value=0).astype("int64")

    # verwijder extreem lange 0-streaks
    s_kept = drop_long_zero_streaks(s)

    # mediaan over ALLE gehouden dagen (inclusief 0)
    med_all = float(s_kept.median()) if len(s_kept) else 0.0
    mean_all = float(s_kept.mean()) if len(s_kept) else 0.0
    active_days = int((s_kept > 0).sum())

    rows.append({
        "donor_id": donor,
        "days_in_window": len(s),
        "days_kept": len(s_kept),
        "zero_days_dropped": int((s == 0).sum() - (s_kept == 0).sum()),
        "median_sessions_per_day": med_all,
        "mean_sessions_per_day": mean_all,
        "active_day_share": (active_days / len(s_kept)) if len(s_kept) else 0.0,
        "category": to_category(med_all),
    })

out = pd.DataFrame(rows)
out["survey_question"] = "q02_sessions_typical_day"
out["gap_minutes"] = GAP_MIN
out["recent_days"] = RECENT_DAYS
out["zero_streak_max"] = ZERO_STREAK_MAX
out["timestamp_utc"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# 6) Schrijven
OUT_XLS.parent.mkdir(exist_ok=True, parents=True)
mode = "a" if OUT_XLS.exists() else "w"
writer_kwargs = {"mode": mode, "engine": "openpyxl"}
if mode == "a":
    writer_kwargs["if_sheet_exists"] = "overlay"
with pd.ExcelWriter(OUT_XLS, **writer_kwargs) as xls:
    out.to_excel(xls, index=False, header=(mode == "w"))

print(f"✅ Q8 geschreven → {OUT_XLS} (n_donors={len(out)})")