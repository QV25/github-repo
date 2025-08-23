#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd

NAMES = ["validation_master_overall.csv", "validation_master_perlabel.csv"]

def find_path(name: str) -> Path|None:
    for p in [Path(name), Path("data")/name]:
        if p.exists(): return p
    return None

def read_smart(path: Path) -> pd.DataFrame:
    # auto-detect delimiter; prefer python engine for sep=None
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", low_memory=False)
    except Exception:
        # fallback tries
        for sep in [";", ",", "\t"]:
            try:
                return pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
            except Exception:
                pass
        raise
    return df

def must_have(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"[ERR] {name}: missing columns: {missing}\nPresent: {list(df.columns)}")

def main():
    overall_path = find_path(NAMES[0])
    perlab_path  = find_path(NAMES[1])
    if not overall_path or not perlab_path:
        sys.exit(f"[ERR] input CSVs not found.\nTried ./{NAMES[0]}, ./data/{NAMES[0]} and ./{NAMES[1]}, ./data/{NAMES[1]}")

    print(f"[INFO] overall: {overall_path}")
    overall = read_smart(overall_path)
    print(f"[INFO] perlabel: {perlab_path}")
    perlab  = read_smart(perlab_path)

    must_have(overall, [
        "question","n",
        "subset_acc","subset_ci_lo","subset_ci_hi",
        "overlap_acc","overlap_ci_lo","overlap_ci_hi",
        "macro_f1","micro_prec","micro_rec","micro_f1"
    ], "overall")

    must_have(perlab, [
        "question","label","tp","fp","fn","precision","recall","f1"
    ], "perlabel")

    print("== overall sample ==")
    print(overall.head(3).to_string(index=False))
    print("\n== perlabel sample ==")
    print(perlab.head(3).to_string(index=False))
    print("\n[OK] inputs readable. Proceed to build tables & figures.")

if __name__ == "__main__":
    main()

