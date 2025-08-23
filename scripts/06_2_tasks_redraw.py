import pathlib, json, re, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER  = ROOT / "derived"
RES  = ROOT / "results"
TAB  = RES / "tables"
FIG  = RES / "figures"
for p in [TAB, FIG]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def slug(s: str):
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')[:60]

def wilson_ci(k, n, alpha=0.05):
    if n == 0: return (np.nan, np.nan)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2)
    p = k / n
    den = 1 + z*z/n
    center = p + z*z/(2*n)
    pm = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    lo = (center - pm) / den
    hi = (center + pm) / den
    return (max(0.0, lo), min(1.0, hi))

def write_latex_table(df: pd.DataFrame, caption: str, label: str, path: pathlib.Path, colfmt=None, index=False):
    body = df.to_latex(index=index, escape=True, longtable=False, bold_rows=False,
                       na_rep="", float_format="%.3f", column_format=colfmt, buf=None)
    wrapped = ["\\begin{table}[t]","\\centering",f"\\caption{{{caption}}}",f"\\label{{{label}}}","\\small",body,"\\end{table}"]
    path.write_text("\n".join(wrapped), encoding="utf-8")

# ---------- load ----------
S = pd.read_parquet(DER / "S_all.parquet")
codebook = pd.read_csv(DER / "tasks_codebook_final.csv")  # question, canonical, regex

# filter op Q12 (main tasks)
cb12 = codebook[codebook["question"]=="Q12"].copy()
cb12["slug"] = cb12["canonical"].apply(slug)
# kolomnamen zoals gemaakt in stap 2: Q12_<slug>
cb12["col"] = "Q12_" + cb12["slug"]

# houd alleen bestaande dummies
cb12 = cb12[cb12["col"].isin(S.columns)].copy()
if cb12.empty:
    raise SystemExit("Geen Q12_* dummies gevonden in S_all.parquet. Run stap 2 eerst.")

# compute shares + Wilson CI
rows = []
N = len(S)
for _, r in cb12.iterrows():
    col = r["col"]
    k = int(pd.to_numeric(S[col], errors="coerce").fillna(0).sum())
    p = k / N if N else np.nan
    lo, hi = wilson_ci(k, N) if N else (np.nan, np.nan)
    rows.append({"Option": r["canonical"], "n": k, "N": N, "share": p, "ci_low": lo, "ci_high": hi})

df = pd.DataFrame(rows).sort_values("share", ascending=False)

# ---------- Figure: dot-plot (sorted, with CI and % labels) ----------
fig, ax = plt.subplots(figsize=(10, max(3.5, 0.5*len(df))))
y = np.arange(len(df))[::-1]
ax.hlines(y, df["ci_low"], df["ci_high"], linewidth=2)
ax.plot(df["share"], y, "o")
ax.set_yticks(y)
ax.set_yticklabels(df["Option"])
ax.set_xlabel("Share selecting this task")
ax.set_xlim(0, min(1.0, (df["ci_high"].max() if len(df) else 1.0)*1.05))
ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.8)
ax.set_title("Q12 — main tasks (per option, Wilson 95% CI)", fontsize=11)

# annotate percentages at points
for xi, yi in zip(df["share"], y):
    if pd.notna(xi):
        ax.text(xi + 0.02, yi, f"{xi*100:.1f}%", va="center", fontsize=9)

fig.tight_layout()
(figpath := FIG / "fig_6_2_tasks.pdf")
fig.savefig(figpath, bbox_inches="tight")
plt.close(fig)

# ---------- Table: tidy shares ----------
out = df.copy()
out["%"] = (out["share"]*100).round(1)
out["CI low %"]  = (out["ci_low"]*100).round(1)
out["CI high %"] = (out["ci_high"]*100).round(1)
out = out[["Option","n","N","%","CI low %","CI high %"]]

(tex_tasks := TAB / "H6_2_tasks_options.tex")
write_latex_table(out, caption="Main tasks (Q12): share per option with Wilson 95\\% CI.",
                  label="tab:6_2_tasks_options", path=tex_tasks, colfmt="l r r r r r", index=False)

# ensure wrapper exists (overwrite with same filename)
(TAB / "H6_2_fig_tasks.tex").write_text(
    "\\begin{figure}[t]\\centering\\includegraphics[width=0.95\\linewidth]{figures/fig_6_2_tasks.pdf}"
    "\\caption{Main tasks (Q12): share per option with Wilson 95\\% CI.}"
    "\\label{fig:6_2_tasks}\\end{figure}\n",
    encoding="utf-8"
)

print("Written:")
print("-", figpath)
print("-", tex_tasks)
print("-", TAB / "H6_2_fig_tasks.tex")

