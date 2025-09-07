# build_h7_figures.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

CSV_OVERALL = "validation_master_overall.csv"
CSV_PERLAB  = "validation_master_perlabel.csv"

# vaste labelvolgorde voor alle per-label figuren
LABEL_ORDER = ["WRI", "BRA", "COD", "LAN", "STU", "OTH"]

# ---------- path & io ----------
def project_paths(name: str) -> list[Path]:
    here = Path(__file__).resolve().parent
    root = here.parent
    return [Path(name), Path("data")/name, here/name, root/name, root/"data"/name, Path.cwd()/name, Path.cwd().parent/name]

def find_path(name: str) -> Path:
    for p in project_paths(name):
        if p.exists(): return p
    raise SystemExit(f"[ERR] not found: {name} (searched: {', '.join(map(str, project_paths(name)))}).")

def read_smart(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", low_memory=False)
    except Exception:
        for sep in [";", ",", "\t"]:
            try: return pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
            except Exception: pass
        raise

def numify(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def normalize_perlabel(df: pd.DataFrame) -> pd.DataFrame:
    # map eventueel oude kolomnamen -> nieuwe
    ren = {}
    if "macro_f1" in df.columns and "precision" not in df.columns: ren["macro_f1"] = "precision"
    if "macro_rec" in df.columns and "recall"   not in df.columns: ren["macro_rec"] = "recall"
    if "micro_f1" in df.columns and "f1"       not in df.columns: ren["micro_f1"]  = "f1"
    df = df.rename(columns=ren)
    for c in ["tp","fp","fn","precision","recall","f1","support"]:
        if c in df.columns: df[c] = numify(df[c])
    # strip spaties uit question/label
    if "question" in df.columns: df["question"] = df["question"].astype(str).str.strip()
    if "label"    in df.columns: df["label"]    = df["label"].astype(str).str.strip()
    return df

def order_labels_fixed(sub: pd.DataFrame) -> pd.DataFrame:
    # labelvolgorde: vaste lijst, daarna eventuele extra labels die niet in LABEL_ORDER staan
    labels_present = list(sub["label"].unique())
    extras = [l for l in labels_present if l not in LABEL_ORDER]
    full_order = LABEL_ORDER + extras
    # reindex op vaste volgorde (alleen aanwezige labels)
    idx_order = [l for l in full_order if l in labels_present]
    return sub.set_index("label").reindex(idx_order).reset_index()

ROOT = Path(__file__).resolve().parent.parent
OUT_FIGS = ROOT/"figures"
OUT_FIGS.mkdir(parents=True, exist_ok=True)

overall = read_smart(find_path(CSV_OVERALL))
perlab  = normalize_perlabel(read_smart(find_path(CSV_PERLAB)))

# sanity
if "micro_f1" not in overall.columns:
    raise SystemExit("[ERR] 'micro_f1' column not found in overall CSV.")
req = {"label","tp","fp","fn","precision","recall","f1","question"}
if not req.issubset(perlab.columns):
    raise SystemExit(f"[ERR] perlabel CSV missing required columns: {sorted(req - set(perlab.columns))}")

# sorteer vragen logisch (Q-nummer)
overall["question"] = overall["question"].astype(str).str.strip()
overall["__o"] = overall["question"].str.extract(r"Q(\d+)", expand=False).astype(float)
overall = overall.sort_values(["__o","question"]).drop(columns="__o")

# --- Micro-F1 by question ---
plt.figure()
x = overall["question"].tolist()
y = numify(overall["micro_f1"]).tolist()
plt.bar(x, y)
plt.ylabel("Micro F1")
plt.xlabel("Question")
plt.ylim(0, 1)
plt.title("Micro-F1 by question")
plt.tight_layout()
plt.savefig(OUT_FIGS/"fig_H7_microF1.pdf")
plt.close()

# --- Per question figures ---
for q in perlab["question"].unique():
    sub = perlab[perlab["question"]==q].copy()
    sub = order_labels_fixed(sub)  # <<< vaste labelvolgorde

    # F1 per label (vaste x-as volgorde)
    plt.figure(figsize=(max(6, 0.55*len(sub)), 4))
    plt.bar(sub["label"], numify(sub["f1"]))
    plt.ylabel("F1")
    plt.xlabel("Label")
    plt.ylim(0, 1)
    plt.title(f"{q}: F1 per label")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_FIGS/f"fig_H7_{q}_f1.pdf")
    plt.close()

    # Stacked TP/FP/FN (zelfde vaste volgorde)
    plt.figure(figsize=(max(6, 0.55*len(sub)), 4))
    labels = sub["label"].tolist()
    tp = numify(sub["tp"]).to_numpy()
    fp = numify(sub["fp"]).to_numpy()
    fn = numify(sub["fn"]).to_numpy()
    ind = np.arange(len(labels))
    width = 0.8
    plt.bar(ind, tp, width, label="TP")
    plt.bar(ind, fp, width, bottom=tp, label="FP")
    plt.bar(ind, fn, width, bottom=tp+fp, label="FN")
    plt.xticks(ind, labels, rotation=35, ha="right")
    plt.ylabel("Count")
    plt.xlabel("Label")
    plt.title(f"{q}: TP/FP/FN per label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIGS/f"fig_H7_{q}_tpfpfn.pdf")
    plt.close()

print(f"Figures written to {OUT_FIGS}")
