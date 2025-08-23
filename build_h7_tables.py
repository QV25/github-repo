# build_h7_tables.py  — robust per-label selection + no-Jinja2 fallback
import pandas as pd
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

def latex_escape(s):
    s = "" if pd.isna(s) else str(s)
    rep = {"\\":"\\textbackslash{}", "&":"\\&", "%":"\\%", "$":"\\$", "#":"\\#",
           "_":"\\_", "{":"\\{", "}":"\\}", "~":"\\textasciitilde{}", "^":"\\textasciicircum{}"}
    return "".join(rep.get(ch, ch) for ch in s)

def to_latex_manual(df: pd.DataFrame, colfmt: str) -> str:
    cols = [latex_escape(c) for c in df.columns]
    lines = [f"\\begin{{tabular}}{{{colfmt}}}", " " + " & ".join(cols) + " \\\\ \\hline"]
    for _, row in df.iterrows():
        vals = [latex_escape(v) for v in row.tolist()]
        lines.append(" " + " & ".join(vals) + " \\\\")
    lines.append("\\end{tabular}")
    return "\n".join(lines)

OVERALL_CSV = find_path("validation_master_overall.csv")
PERLABEL_CSV = find_path("validation_master_perlabel.csv")

out_tables = Path("tables")
out_tables.mkdir(parents=True, exist_ok=True)

overall = read_smart(OVERALL_CSV)
perlab  = read_smart(PERLABEL_CSV)

# ---------- OVERALL ----------
ov = overall.copy()
num_cols = ["n","subset_acc","subset_ci_lo","subset_ci_hi","overlap_acc",
            "overlap_ci_lo","overlap_ci_hi","macro_f1","micro_prec","micro_rec","micro_f1"]
for c in num_cols:
    if c in ov.columns: ov[c] = numify(ov[c])

order = {f"Q{i}": i for i in range(1, 100)}
ov["__ord__"] = ov["question"].map(order).fillna(9999)
ov = ov.sort_values(["__ord__","question"]).drop(columns="__ord__")

fmt2 = lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
for c in ["subset_acc","subset_ci_lo","subset_ci_hi","overlap_acc",
          "overlap_ci_lo","overlap_ci_hi","macro_f1","micro_prec","micro_rec","micro_f1"]:
    if c in ov.columns: ov[c] = ov[c].map(fmt2)

ov["Subset acc (95% CI)"]  = ov.apply(lambda r: f"{r['subset_acc']} [{r['subset_ci_lo']}, {r['subset_ci_hi']}]", axis=1)
ov["Any-overlap (95% CI)"] = ov.apply(lambda r: f"{r['overlap_acc']} [{r['overlap_ci_lo']}, {r['overlap_ci_hi']}]", axis=1)

ov_tab = ov[["question","n","Subset acc (95% CI)","Any-overlap (95% CI)","macro_f1","micro_prec","micro_rec","micro_f1"]] \
           .rename(columns={"question":"Question","n":"N","macro_f1":"Macro F1",
                            "micro_prec":"Micro Prec","micro_rec":"Micro Rec","micro_f1":"Micro F1"})

out_master = out_tables/"H7_master_overall.tex"
try:
    latex = ov_tab.to_latex(index=False, escape=True, column_format="l r l l r r r r")
except Exception:
    latex = to_latex_manual(ov_tab, "l r l l r r r r")
out_master.write_text(latex)
print("Wrote:", out_master)

# ---------- PER-LABEL ----------
pl = perlab.copy()
for c in ["tp","fp","fn","precision","recall","f1","support"]:
    if c in pl.columns: pl[c] = numify(pl[c])

for q in pl["question"].unique():
    df = pl[pl["question"]==q].copy()

    # format metrics BEFORE rename
    if "precision" in df.columns: df["precision"] = df["precision"].map(fmt2)
    if "recall"   in df.columns: df["recall"]   = df["recall"].map(fmt2)
    if "f1"       in df.columns: df["f1"]       = df["f1"].map(fmt2)

    has_supp = "support" in df.columns

    # rename to presentation headers
    df = df.rename(columns={"label":"Label","tp":"TP","fp":"FP","fn":"FN",
                            "precision":"Precision","recall":"Recall","f1":"F1",
                            "support":"Support"})

    # sort by F1 desc if present
    if "F1" in df.columns:
        s = pd.to_numeric(df["F1"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        df = df.assign(__s=s).sort_values("__s", ascending=False).drop(columns="__s")

    display_cols = ["Label","TP","FP","FN","Precision","Recall","F1"] + (["Support"] if has_supp else [])
    colfmt       = "l r r r r r r" + (" r" if has_supp else "")

    outp = out_tables/f"H7_{q}_perlabel.tex"
    try:
        outp.write_text(df[display_cols].to_latex(index=False, escape=True, column_format=colfmt))
    except Exception:
        outp.write_text(to_latex_manual(df[display_cols], colfmt))
    print("Wrote:", outp)

print("LaTeX tables written to ./tables")

