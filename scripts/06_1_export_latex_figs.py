import pathlib, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES  = ROOT / "results"
TAB  = RES / "tables"
FIG  = RES / "figures"
DER  = ROOT / "derived"
for p in [TAB, FIG]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- load inputs from 6.1 ----------
funnel_csv  = TAB / "table_6_1_funnel.csv"
balance_csv = TAB / "table_6_1_balance_smd.csv"
if not funnel_csv.exists() or not balance_csv.exists():
    raise SystemExit("Run scripts/06_1_funnel_balance.py first to create the CSVs.")

funnel  = pd.read_csv(funnel_csv)
balance = pd.read_csv(balance_csv)

# ============ LaTeX tables ============
def write_latex_table(df: pd.DataFrame, caption: str, label: str, path: pathlib.Path, colfmt=None, index=False):
    latex_body = df.to_latex(index=index, escape=True, longtable=False, bold_rows=False,
                             na_rep="", float_format="%.3f", column_format=colfmt, buf=None)
    wrapped = []
    wrapped.append("\\begin{table}[t]")
    wrapped.append("\\centering")
    wrapped.append("\\caption{" + caption + "}")
    wrapped.append("\\label{" + label + "}")
    wrapped.append("\\small")
    wrapped.append(latex_body)
    wrapped.append("\\end{table}")
    path.write_text("\n".join(wrapped), encoding="utf-8")

# Funnel LaTeX
funnel_tex = TAB / "H6_1_funnel.tex"
write_latex_table(
    funnel.rename(columns={"step":"Step","n":"N"}),
    caption="Sample funnel for Chapter 6.",
    label="tab:6_1_funnel",
    path=funnel_tex,
    colfmt="l r"
)

# Balance SMD LaTeX (rounded)
bal_show = balance.copy()
for c in ["SMD","absSMD"]:
    if c in bal_show.columns:
        bal_show[c] = bal_show[c].astype(float).round(3)
bal_show.rename(columns={"variable":"Variable","level":"Level","absSMD":"|SMD|"}, inplace=True)
balance_tex = TAB / "H6_1_balance_smd.tex"
write_latex_table(
    bal_show[["Variable","Level","SMD","|SMD|"]],
    caption="Balance between donors and non-donors: standardized mean differences (SMD) by level.",
    label="tab:6_1_balance_smd",
    path=balance_tex,
    colfmt="l l r r"
)

# Also create tiny figure-include tex helpers
(fig_funnel_tex := TAB / "H6_1_fig_funnel.tex").write_text(
    "\\begin{figure}[t]\n\\centering\n"
    "\\includegraphics[width=0.85\\linewidth]{figures/fig_6_1_funnel.pdf}\n"
    "\\caption{Funnel diagram for Chapter 6.}\\label{fig:6_1_funnel}\n\\end{figure}\n",
    encoding="utf-8"
)
(fig_love_tex := TAB / "H6_1_fig_loveplot.tex").write_text(
    "\\begin{figure}[t]\n\\centering\n"
    "\\includegraphics[width=0.9\\linewidth]{figures/fig_6_1_loveplot.pdf}\n"
    "\\caption{Absolute standardized mean differences (|SMD|) across levels.}"
    "\\label{fig:6_1_loveplot}\n\\end{figure}\n",
    encoding="utf-8"
)

# ============ Figures ============
# 1) Funnel diagram (simple boxes + arrows)
def draw_funnel(df: pd.DataFrame, savepath_pdf: pathlib.Path, savepath_png: pathlib.Path):
    steps = df["step"].tolist()
    ns    = df["n"].tolist()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")

    y0 = 0.85
    dy = 0.28
    box_w = 0.78
    box_h = 0.12

    for i, (s, n) in enumerate(zip(steps, ns)):
        y = y0 - i*dy
        # box
        ax.add_patch(plt.Rectangle((0.11, y-box_h/2), box_w, box_h, fill=False, linewidth=1.5))
        ax.text(0.5, y+0.02, s, ha="center", va="center", fontsize=11)
        ax.text(0.5, y-0.04, f"N = {n}", ha="center", va="center", fontsize=10)
        # arrow to next
        if i < len(steps)-1:
            ax.annotate("", xy=(0.5, y - box_h/2 - 0.02), xytext=(0.5, y - dy + box_h/2 + 0.02),
                        arrowprops=dict(arrowstyle="->", lw=1.5))

    fig.tight_layout()
    fig.savefig(savepath_pdf, bbox_inches="tight")
    fig.savefig(savepath_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

# 2) Love plot (abs SMD across all levels, sorted)
def draw_loveplot(bal: pd.DataFrame, savepath_pdf: pathlib.Path, savepath_png: pathlib.Path, top_n=20):
    b = bal.copy()
    b["absSMD"] = b["SMD"].astype(float).abs()
    # optional: drop '<not found>' if present
    b = b[~b["level"].astype(str).str.contains("<not found>", na=False)]
    b = b.sort_values("absSMD", ascending=True)
    if len(b) > top_n:
        b = b.iloc[-top_n:]  # keep top_n
    labels = (b["variable"] + " — " + b["level"].astype(str)).tolist()
    y = np.arange(len(b))

    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.4*len(b))))
    ax.hlines(y, 0, b["absSMD"], linewidth=2)
    ax.plot(b["absSMD"], y, "o")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("|SMD|")
    ax.set_xlim(left=0)
    ax.grid(axis="x", linestyle=":", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(savepath_pdf, bbox_inches="tight")
    fig.savefig(savepath_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

# draw & save
funnel_pdf = FIG / "fig_6_1_funnel.pdf"
funnel_png = FIG / "fig_6_1_funnel.png"
love_pdf   = FIG / "fig_6_1_loveplot.pdf"
love_png   = FIG / "fig_6_1_loveplot.png"

draw_funnel(funnel, funnel_pdf, funnel_png)
draw_loveplot(balance, love_pdf, love_png, top_n=20)

print("Written:")
print("-", funnel_tex)
print("-", balance_tex)
print("-", fig_funnel_tex)
print("-", fig_love_tex)
print("-", funnel_pdf)
print("-", love_pdf)


