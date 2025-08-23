#!/usr/bin/env python3
import argparse, json, re, sys, random
from pathlib import Path
import pandas as pd

# --------- helpers ----------
NOT_CHOSEN_PAT = re.compile(r"\bi\s*did\s*not\s*choose\b", re.I)

# veldkandidaten voor donor-id in logs/jsonl
LOGS_ID_CANDS = ["donor_id","donor","donor_hash","Response ID","ResponseId","id"]
JSONL_ID_CANDS = ["donor_id","donor","donor_hash","ResponseId","id"]

# doelkolom-fragmenten (case-insensitive) om de juiste LOGS-kolom te vinden
QCOL_HINTS = {
  13: ["if you chose", "writing", "professional", "sub-tasks"],
  14: ["if you chose", "brainstorming", "personal ideas", "fun"],
  15: ["if you chose", "coding", "programming", "sub"],
  16: ["if you chose", "language practice", "translation"],
  17: ["if you chose", "study revision", "exam prep", "study tasks"],
}

def norm(s: str) -> str:
    s = re.sub(r"[“”\"']", "", str(s))
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s

def find_logs_id_col(df: pd.DataFrame, override: str|None):
    if override and override in df.columns: return override
    for c in LOGS_ID_CANDS:
        if c in df.columns: return c
    return None

def find_jsonl_id_col(df: pd.DataFrame, override: str|None):
    if override and override in df.columns: return override
    for c in JSONL_ID_CANDS:
        if c in df.columns: return c
    return None

def find_qcol(df: pd.DataFrame, q: int, override: str|None):
    if override and override in df.columns: return override
    hints = [h.lower() for h in QCOL_HINTS[q]]
    # exacte match eerst
    for c in df.columns:
        if all(h in norm(c) for h in hints):
            return c
    # fallback: kolom die start met "if you chose" en >=2 hints matcht
    for c in df.columns:
        nc = norm(c)
        if nc.startswith("if you chose") and sum(h in nc for h in hints) >= 2:
            return c
    return None

def read_logs(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, header=1, low_memory=False)
    except Exception:
        return pd.read_csv(path, header=0, low_memory=False)

def build_whitelist(logs_csv: Path, q: int, logs_id_col=None, logs_qcol=None) -> set[str]:
    logs = read_logs(logs_csv)
    id_col = find_logs_id_col(logs, logs_id_col)
    if id_col is None:
        sys.exit(f"[ERR] kon donor-ID kolom niet vinden in {logs_csv}. Beschikbare kolommen: {list(logs.columns)[:20]}")
    qcol = find_qcol(logs, q, logs_qcol)
    if qcol is None:
        sys.exit(f"[ERR] kon Q{q}-kolom niet vinden in {logs_csv}. Geef desnoods --logs-qcol op.")

    donors = set()
    ser = logs[qcol].astype(str)
    for i, val in ser.items():
        if not isinstance(val, str) or val.strip()=="":
            continue
        # multi-select is ;-gescheiden
        toks = [t.strip() for t in val.replace("；",";").split(";") if t.strip()]
        if any(not NOT_CHOSEN_PAT.search(t) for t in toks):
            donors.add(str(logs.at[i, id_col]))
    if not donors:
        sys.exit(f"[ERR] geen nuttige donors voor Q{q} in {logs_csv}")
    return donors

def load_jsonl_pool(jsonl: Path, jsonl_id_override: str|None):
    rows=[]
    with jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            try: o=json.loads(line)
            except Exception: continue
            rows.append(o)
    df = pd.DataFrame(rows)
    id_col = find_jsonl_id_col(df, jsonl_id_override)
    if id_col is None:
        sys.exit(f"[ERR] donor-id kolom niet gevonden in {jsonl}. Kolommen: {list(df.columns)[:20]}")
    # rename naar donor_id voor eenvoud
    if id_col != "donor_id":
        df = df.rename(columns={id_col:"donor_id"})
    # selecteer Q/A velden
    keep = ["donor_id"]
    for cand in ["question","prompt","user_text"]:
        if cand in df.columns: keep.append(cand); break
    for cand in ["answer","assistant","assistant_text","reply"]:
        if cand in df.columns: keep.append(cand); break
    if len(keep) < 3:
        sys.exit(f"[ERR] kon question/answer velden niet vinden in {jsonl}.")
    df = df[keep].rename(columns={keep[1]:"question_text", keep[2]:"answer_text"})
    df = df[(df["question_text"].astype(str).str.strip()!="") & (df["answer_text"].astype(str).str.strip()!="")]
    df = df.drop_duplicates(subset=["donor_id","question_text","answer_text"])
    return df

def sample_random_from_whitelist(df_pool: pd.DataFrame, wl: set[str],
                                 n_samples=100, min_donors=4, max_per_donor=25, seed=2025) -> pd.DataFrame:
    rng = random.Random(seed)
    dfw = df_pool[df_pool["donor_id"].astype(str).isin({str(x) for x in wl})].copy()
    if dfw.empty:
        sys.exit("[ERR] whitelist donors komen niet voor in parsed/all.jsonl (donor_id mismatch).")
    donors = dfw["donor_id"].value_counts().index.tolist()
    donors = donors[:max(min_donors, len(donors))]
    groups = {d: dfw[dfw["donor_id"]==d].sample(frac=1, random_state=rng.randint(1,10_000)).reset_index(drop=True)
              for d in donors}
    idx={d:0 for d in donors}; take={d:0 for d in donors}; picks=[]
    # ronde 1: cap per donor
    while len(picks) < n_samples and any(idx[d] < len(groups[d]) for d in donors):
        for d in donors:
            if len(picks) >= n_samples: break
            if take[d] >= max_per_donor: continue
            i = idx[d]
            if i < len(groups[d]):
                picks.append(groups[d].iloc[i]); idx[d]=i+1; take[d]+=1
    # ronde 2: zonder cap
    if len(picks) < n_samples:
        rem = pd.concat([groups[d].iloc[idx[d]:] for d in donors], ignore_index=True)
        need = min(n_samples - len(picks), len(rem))
        if need > 0:
            picks += rem.sample(n=need, random_state=rng.randint(1,10_000)).to_dict("records")
    out = pd.DataFrame(picks[:n_samples])
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    out.insert(0, "item_id", out.index+1)
    return out[["item_id","donor_id","question_text","answer_text"]]

def main():
    ap = argparse.ArgumentParser(description="Build random Q+A sample per question based on donor whitelist from Data logs.csv")
    ap.add_argument("--q", type=int, required=True, choices=[13,14,15,16,17], help="question number (13..17)")
    ap.add_argument("--logs", default="data/Data logs.csv", help="path to Data logs.csv")
    ap.add_argument("--jsonl", default="parsed/all.jsonl", help="path to parsed/all.jsonl")
    ap.add_argument("--out", default=None, help="output CSV path (default qXX_sample_100.csv)")
    ap.add_argument("--logs-id-col", default=None, help="override donor id column name in logs csv")
    ap.add_argument("--jsonl-id-col", default=None, help="override donor id column name in jsonl")
    ap.add_argument("--min-donors", type=int, default=4)
    ap.add_argument("--max-per-donor", type=int, default=25)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    logs = Path(args.logs); jsonl = Path(args.jsonl)
    out = Path(args.out) if args.out else Path(f"q{args.q}_sample_100.csv")

    wl = build_whitelist(logs, args.q, args.logs_id_col, None)
    pool = load_jsonl_pool(jsonl, args.jsonl_id_col)
    sample = sample_random_from_whitelist(pool, wl, n_samples=100,
                                          min_donors=args.min_donors, max_per_donor=args.max_per_donor, seed=args.seed)
    sample.to_csv(out, index=False, encoding="utf-8")
    print(f"[DONE] {out.resolve()}  rows={len(sample)}  donors={sample['donor_id'].nunique()}")

if __name__ == "__main__":
    main()

