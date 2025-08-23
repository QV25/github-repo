# build_h7_figures.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def find_path(name: str) -> Path:
    for p in [Path(name), Path("data")/name]:
        if p.exists(): return p
    raise SystemExit(f"[ERR] not found: {name} (tried ./{name} and ./data/{name})")

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

OVERALL_CSV = find_path("validation_master_overall.csv")
PERLABEL_CSV = find_path("validation_master_perlabel.csv")

out_figs = Path("figures")
out_figs.mkdir(parents=True, exist_ok=True)

overall = read_smart(OVERALL_CSV)
perlab  = read_smart(PERLABEL_CSV)

if "micro_f1" not in overall.columns:
    raise SystemExit("[ERR] 'micro_f1' column not found in overall CSV (delimiter fixed?).")

overall["micro_f1"] = numify(overall["micro_f1"])
for c in ["tp","fp","fn","precision","recall","f1"]:
    if c in perlab.columns:
        perlab[c] = numify(perlab[c])

# --- Micro-F1 by question ---
plt.figure()
order_map = {f"Q{i}": i for i in range(1,60)}
overall["__o"] = overall["question"].map(order_map).fillna(999)
overall = overall.sort_values(["__o","question"]).drop(columns="__o")
x = overall["question"].tolist()
y = overall["micro_f1"].tolist()
plt.bar(x, y)
plt.ylabel("Micro F1")
plt.xlabel("Question")
plt.ylim(0, 1)
plt.title("Micro-F1 by question")
plt.tight_layout()
plt.savefig(out_figs/"fig_H7_microF1.pdf")
plt.close()

# --- Per question figures ---
for q in perlab["question"].unique():
    sub = perlab[perlab["question"]==q].copy()

    # F1 per label
    plt.figure(figsize=(max(6, 0.55*len(sub)), 4))
    order = sub.sort_values("f1", ascending=False)
    plt.bar(order["label"], order["f1"])
    plt.ylabel("F1")
    plt.xlabel("Label")
    plt.ylim(0, 1)
    plt.title(f"{q}: F1 per label")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_figs/f"fig_H7_{q}_f1.pdf")
    plt.close()

    # Stacked TP/FP/FN
    plt.figure(figsize=(max(6, 0.55*len(sub)), 4))
    labels = sub["label"].tolist()
    tp = sub["tp"].to_numpy()
    fp = sub["fp"].to_numpy()
    fn = sub["fn"].to_numpy()
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
    plt.savefig(out_figs/f"fig_H7_{q}_tpfpfn.pdf")
    plt.close()

print("Figures written to ./figures")

