#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np

CSV_OVERALL = "validation_master_overall.csv"
CSV_PERLAB  = "validation_master_perlabel.csv"

# ---------- utilities ----------
def project_paths(name: str) -> list[Path]:
    """Try common locations: scripts/, project root, project root/data, CWD, parent of CWD."""
    here = Path(__file__).resolve().parent
    root = here.parent
    return [
        Path(name),
        Path("data")/name,
        here/name,
        root/name,
        root/"data"/name,
        Path.cwd()/name,
        Path.cwd().parent/name,
    ]

def find_path(name: str) -> Path:
    for p in project_paths(name):
        if p.exists():
            return p
    raise SystemExit(f"[ERR] not found: {name} (searched: {', '.join(map(str, project_paths(name)))}).")

def read_smart(path: Path) -> pd.DataFrame:
    # robust CSV reader (auto-detect ; , \t)
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", low_memory=False)
    except Exception:
        for sep in [";", ",", "\t"]:
            try:
                return pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
            except Exception:
                pass
        raise

def numify(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def normalize_perlabel(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy column names to the canonical ones and cast numerics."""
    ren = {}
    # legacy → canonical
    if "macro_f1" in df.columns and "precision" not in df.columns:
        ren["macro_f1"] = "precision"
    if "macro_rec" in df.columns and "recall" not in df.columns:
        ren["macro_rec"] = "recall"
    if "micro_f1" in df.columns and "f1" not in df.columns:
        ren["micro_f1"] = "f1"
    df = df.rename(columns=ren)

    # must-have columns
    need = ["question","label","tp","fp","fn","precision","recall","f1"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"[ERR] perlabel missing columns: {missing}\nPresent: {list(df.columns)}")

    # numeric casting
    for c in ["tp","fp","fn","precision","recall","f1","support"]:
        if c in df.columns:
            df[c] = numify(df[c])
    return df

def normalize_overall(df: pd.DataFrame) -> pd.DataFrame:
    # must-haves present in your files
    need_min = ["question","n","overlap_acc","overlap_ci_lo","overlap_ci_hi","macro_f1","macro_rec","micro_f1"]
    missing = [c for c in need_min if c not in df.columns]
    if missing:
        sys.exit(f"[ERR] overall missing columns: {missing}\nPresent: {list(df.columns)}")

    # numeric casting
    num_cols = ["n","overlap_acc","overlap_ci_lo","overlap_ci_hi","macro_f1","macro_rec","micro_f1",
                "subset_acc","subset_ci_lo","subset_ci_hi","micro_prec","micro_rec"]
    for c in num_cols:
        if c in df.columns:
            df[c] = numify(df[c])
    return df

# ---------- main ----------
def main():
    overall_path = find_path(CSV_OVERALL)
    perlabel_path = find_path(CSV_PERLAB)

    print(f"[INFO] overall: {overall_path}")
    overall = read_smart(overall_path)
    overall = normalize_overall(overall)

    print(f"[INFO] perlabel: {perlabel_path}")
    perlab = read_smart(perlabel_path)
    perlab = normalize_perlabel(perlab)

    # recompute pooled micro metrics from perlabel TP/FP/FN
    agg = perlab.groupby("question")[["tp","fp","fn"]].sum().reset_index()
    agg["micro_prec_calc"] = agg["tp"] / (agg["tp"] + agg["fp"])
    agg["micro_rec_calc"]  = agg["tp"] / (agg["tp"] + agg["fn"])
    agg["micro_f1_calc"]   = 2*agg["micro_prec_calc"]*agg["micro_rec_calc"] / (agg["micro_prec_calc"]+agg["micro_rec_calc"])

    merged = overall.merge(agg[["question","micro_prec_calc","micro_rec_calc","micro_f1_calc"]], on="question", how="left")

    # compare
    print("\n== CHECK: micro metrics (overall vs. recomputed from perlabel) ==")
    rows = []
    for _, r in merged.iterrows():
        q = r["question"]
        o_f1 = r["micro_f1"]; c_f1 = r["micro_f1_calc"]
        o_p  = r.get("micro_prec", np.nan); c_p = r["micro_prec_calc"]
        o_r  = r.get("micro_rec",  np.nan); c_r = r["micro_rec_calc"]
        rows.append([q, o_p, c_p, o_r, c_r, o_f1, c_f1, (o_f1-c_f1 if pd.notna(o_f1) and pd.notna(c_f1) else np.nan)])
    chk = pd.DataFrame(rows, columns=["question","micro_prec(overall)","micro_prec(calc)",
                                      "micro_rec(overall)","micro_rec(calc)",
                                      "micro_f1(overall)","micro_f1(calc)","Δmicro_f1 (overall - calc)"])
    print(chk.to_string(index=False, float_format=lambda v: f"{v:.4f}" if pd.notna(v) else "NA"))

    # lightweight sanity: weighted macro-F1 and balanced accuracy recompute (optional)
    def w_macro_f1(dfq: pd.DataFrame) -> float:
        sup = dfq["tp"] + dfq["fn"]
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = 2*dfq["tp"]/(2*dfq["tp"] + dfq["fp"] + dfq["fn"])
        f1 = f1.fillna(0.0)
        if sup.sum() == 0: return np.nan
        return float((f1*sup).sum()/sup.sum())

    def balanced_acc(dfq: pd.DataFrame) -> float:
        sup = dfq["tp"] + dfq["fn"]
        mask = sup > 0
        if mask.sum() == 0: return np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            rec = (dfq.loc[mask,"tp"] / sup[mask]).fillna(0.0)
        return float(rec.mean())

    print("\n== CHECK: macro summaries (overall vs. perlabel recompute) ==")
    for q, sub in perlab.groupby("question"):
        wm = w_macro_f1(sub)
        ba = balanced_acc(sub)
        ov_row = overall[overall["question"]==q].iloc[0]
        print(f"{q}: macro_f1 overall={ov_row['macro_f1']:.4f} vs calc={wm:.4f} | macro_rec overall={ov_row['macro_rec']:.4f} vs calc={ba:.4f}")

    print("\n[OK] inputs validated. You can now build figures & tables.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
