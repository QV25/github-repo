# build_h7_tables.py — master table EXACTLY matches overall CSV columns (no extras)
import pandas as pd
from pathlib import Path

CSV_OVERALL = "validation_master_overall.csv"
CSV_PERLAB  = "validation_master_perlabel.csv"  # alleen nodig voor per-label tabellen, optioneel

# ---------- path & io ----------
def project_paths(name: str) -> list[Path]:
    here = Path(__file__).resolve().parent
    root = here.parent
    return [Path(name), Path("data")/name, here/name, root/name, root/"data"/name,
            Path.cwd()/name, Path.cwd().parent/name]

def find_path(name: str) -> Path:
    for p in project_paths(name):
        if p.exists(): return p
    raise SystemExit(f"[ERR] not found: {name} (searched: {', '.join(map(str, project_paths(name)))}).")

def read_smart(path: Path) -> pd.DataFrame:
    # robust read for ; , \t with UTF-8 BOM handling
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        for sep in [";", ",", "\t"]:
            try: return pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            except Exception: pass
        raise

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    # strip BOM and spaces from column names
    return df.rename(columns=lambda c: str(c).replace("\ufeff","").strip())

def numify(s: pd.Series) -> pd.Series:
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

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT/"tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- OVERALL MASTER TABLE ----------
ov = read_smart(find_path(CSV_OVERALL))
ov = clean_cols(ov)
# clean values
ov["question"] = ov["question"].astype(str).str.replace("\ufeff","").str.strip()

# enforce numeric + formatting
for c in ["overlap_acc","overlap_ci_lo","overlap_ci_hi","macro_f1","macro_rec","micro_f1","n"]:
    if c in ov.columns: ov[c] = numify(ov[c])

# sort by Q-number
ov["__ord__"] = ov["question"].str.extract(r"Q(\d+)", expand=False).astype(float)
ov = ov.sort_values(["__ord__","question"]).drop(columns="__ord__")

# format to two decimals for presentation
fmt2 = lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
for c in ["overlap_acc","overlap_ci_lo","overlap_ci_hi","macro_f1","macro_rec","micro_f1"]:
    ov[c] = ov[c].map(fmt2)
# n blijft integer-achtig (geen format nodig)

# EXACT column order & headers as requested
columns_exact = ["question","n","overlap_acc","overlap_ci_lo","overlap_ci_hi","macro_f1","macro_rec","micro_f1"]
missing = [c for c in columns_exact if c not in ov.columns]
if missing:
    raise SystemExit(f"[ERR] overall CSV missing required columns: {missing}\nPresent: {list(ov.columns)}")

ov_tab = ov[columns_exact].rename(columns={
    "question":"Question",
    "n":"N",
    "overlap_acc":"Overlap acc",
    "overlap_ci_lo":"Overlap CI lo",
    "overlap_ci_hi":"Overlap CI hi",
    "macro_f1":"Macro F1",
    "macro_rec":"Macro Rec",
    "micro_f1":"Micro F1"
})

# write LaTeX
out_master = OUT_DIR/"H7_master_overall.tex"
try:
    latex = ov_tab.to_latex(index=False, escape=True, column_format="l r r r r r r r")
except Exception:
    latex = to_latex_manual(ov_tab, "l r r r r r r r")
out_master.write_text(latex, encoding="utf-8")
print("Wrote:", out_master)

# ---------- (optional) PER-LABEL TABLES ----------
# Als je per-label tabellen ook wilt genereren, laat dit AAN.
# Zoniet, kun je de hele sectie hieronder verwijderen of achter een flag zetten.
GEN_PERLABEL = True
if GEN_PERLABEL and Path(find_path(CSV_PERLAB)).exists():
    pl = read_smart(find_path(CSV_PERLAB))
    pl = clean_cols(pl)
    for c in ["tp","fp","fn","precision","recall","f1","support"]:
        if c in pl.columns: pl[c] = numify(pl[c])
    pl["question"] = pl["question"].astype(str).str.strip()
    pl["label"]    = pl["label"].astype(str).str.strip()

    # label volgorde vastleggen indien gewenst
    LABEL_ORDER = ["WRI","BRA","COD","LAN","STU","OTH"]
    def order_labels_fixed(dfq: pd.DataFrame) -> pd.DataFrame:
        labs = list(dfq["label"].unique())
        extras = [l for l in labs if l not in LABEL_ORDER]
        idx = [l for l in LABEL_ORDER if l in labs] + extras
        return dfq.set_index("label").reindex(idx).reset_index()

    # format
    pfmt = lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
    for c in ["precision","recall","f1"]:
        if c in pl.columns: pl[c] = pl[c].map(pfmt)

    for q in pl["question"].unique():
        df = pl[pl["question"]==q].copy()
        df = order_labels_fixed(df)
        df = df.rename(columns={"label":"Label","tp":"TP","fp":"FP","fn":"FN",
                                "precision":"Precision","recall":"Recall","f1":"F1","support":"Support"})
        display_cols = ["Label","TP","FP","FN","Precision","Recall","F1"] + (["Support"] if "Support" in df.columns else [])
        colfmt = "l r r r r r r" + (" r" if "Support" in df.columns else "")
        outp = OUT_DIR/f"H7_{q}_perlabel.tex"
        try:
            outp.write_text(df[display_cols].to_latex(index=False, escape=True, column_format=colfmt), encoding="utf-8")
        except Exception:
            outp.write_text(to_latex_manual(df[display_cols], colfmt), encoding="utf-8")
        print("Wrote:", outp)
