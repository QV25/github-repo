import pathlib, json, math, re
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER  = ROOT / "derived"
RES  = ROOT / "results"
TAB  = RES / "tables"
FIG  = RES / "figures"
for p in [TAB, FIG]: p.mkdir(parents=True, exist_ok=True)

# ---------------- helpers ----------------
def load_sets():
    S = pd.read_parquet(DER/"S_donors.parquet")   # donor survey (n≈24)
    L = pd.read_parquet(DER/"L_logs.parquet")     # logs-derived (n≈24)
    labels = json.load(open(DER/"labels_SurveyFull.json", encoding="utf-8"))
    codebook = pd.read_csv(DER/"tasks_codebook_final.csv")  # question, canonical, regex
    return S, L, labels, codebook

def fieldname_for_q(labels, qcode):
    for col, lab in labels.items():
        if str(lab).startswith(qcode):
            return col
    return None

def wilson_ci(k, n, alpha=0.05):
    if n == 0: return (np.nan, np.nan)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2); p = k/n
    den = 1 + z*z/n
    center = p + z*z/(2*n)
    pm = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    lo = (center - pm) / den; hi = (center + pm) / den
    return (max(0.0, lo), min(1.0, hi))

def newcombe_diff_ci(k1, n1, k2, n2, alpha=0.05):
    p1_lo, p1_hi = wilson_ci(k1, n1, alpha)
    p2_lo, p2_hi = wilson_ci(k2, n2, alpha)
    return (p1_lo - p2_hi, p1_hi - p2_lo)

def cliff_delta(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().values
    if len(x)==0 or len(y)==0: return np.nan
    gt = 0; lt = 0
    for xi in x:
        gt += np.sum(xi > y); lt += np.sum(xi < y)
    return (gt - lt) / (len(x)*len(y)) if len(x)*len(y)>0 else np.nan

def hodges_lehmann(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().values
    if len(x)==0 or len(y)==0: return np.nan
    diffs = (x.reshape(-1,1) - y.reshape(1,-1)).ravel()
    return float(np.median(diffs))

def ecdf(arr):
    v = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().sort_values().values
    if len(v)==0: return np.array([]), np.array([])
    y = np.arange(1, len(v)+1)/len(v)
    return v, y

def slug(s: str):
    return re.sub(r'[^A-Za-z0-9]+','_', s).strip('_')[:60]

# --------------- load ---------------
S, L, labels, codebook = load_sets()

# Identify columns for Q7-9 numeric and Q10-11 categorical
Q7_colS, Q8w_colS, Q9_colS = "Q7_mid", "Q8w_mid", "Q9_mid"
Q7_colL, Q8w_colL, Q9_colL = "Q7_mid", "Q8w_mid", "Q9_mid"

Q10_field = fieldname_for_q(labels, "Q10")
Q11_field = fieldname_for_q(labels, "Q11")
Q10_S = Q10_field if Q10_field in S.columns else None
Q11_S = Q11_field if Q11_field in S.columns else None
Q10_L = Q10_field if Q10_field in L.columns else None
Q11_L = Q11_field if Q11_field in L.columns else None

# ---------------- NUMERIC: Q7/Q8/Q9 ----------------
num_rows = []
num_effects = []

for q, (cs, cl, title, unit) in {
    "Q7": (Q7_colS, Q7_colL, "Q7 sessions (last 7 days)", ""),
    "Q8": (Q8w_colS, Q8w_colL, "Q8 sessions/week (from per day)", ""),
    "Q9": (Q9_colS, Q9_colL, "Q9 session duration", "minutes"),
}.items():
    xs = pd.to_numeric(S[cs], errors="coerce")
    xl = pd.to_numeric(L[cl], errors="coerce")
    Ns, Nl = xs.notna().sum(), xl.notna().sum()
    med_s = xs.median(skipna=True); p25_s = xs.quantile(0.25); p75_s = xs.quantile(0.75)
    med_l = xl.median(skipna=True); p25_l = xl.quantile(0.25); p75_l = xl.quantile(0.75)
    try:
        p_u = float(mannwhitneyu(xs.dropna(), xl.dropna(), alternative="two-sided").pvalue)
    except Exception:
        p_u = np.nan
    hl = hodges_lehmann(xs, xl)
    # simple bootstrap CI for HL
    rng = np.random.default_rng(42); B=3000
    hl_samp = []
    xsa, xla = xs.dropna().values, xl.dropna().values
    for _ in range(B):
        xb = rng.choice(xsa, size=len(xsa), replace=True)
        yb = rng.choice(xla, size=len(xla), replace=True)
        hl_samp.append(np.median((xb.reshape(-1,1)-yb.reshape(1,-1)).ravel()))
    hl_lo, hl_hi = float(np.percentile(hl_samp, 2.5)), float(np.percentile(hl_samp, 97.5))
    cd = cliff_delta(xs, xl)

    num_rows.append({
        "Item": q, "Title": title,
        "N_survey": int(Ns), "Median_survey": float(med_s) if med_s==med_s else np.nan, "P25_s": float(p25_s), "P75_s": float(p75_s),
        "N_logs": int(Nl), "Median_logs": float(med_l) if med_l==med_l else np.nan, "P25_l": float(p25_l), "P75_l": float(p75_l),
        "U_p": p_u, "HL_diff": float(hl) if hl==hl else np.nan, "HL_CI_low": hl_lo, "HL_CI_high": hl_hi, "Cliffs_delta": cd,
        "Unit": unit
    })
    num_effects.append({"label": f"{q} (HL median diff)", "effect": float(hl) if hl==hl else np.nan, "lo": hl_lo, "hi": hl_hi})

numtab = pd.DataFrame(num_rows)
# write latex
def write_latex_table(df, caption, label, path, colfmt=None, index=False):
    body = df.to_latex(index=index, escape=True, longtable=False, bold_rows=False,
                       na_rep="", float_format="%.3f", column_format=colfmt, buf=None)
    wrapped = ["\\begin{table}[t]","\\centering",f"\\caption{{{caption}}}",f"\\label{{{label}}}","\\small",body,"\\end{table}"]
    path.write_text("\n".join(wrapped), encoding="utf-8")

show_num = numtab.copy()
for c in ["Median_survey","P25_s","P75_s","Median_logs","P25_l","P75_l","U_p","HL_diff","HL_CI_low","HL_CI_high","Cliffs_delta"]:
    if c in show_num.columns: show_num[c] = show_num[c].astype(float).round(3)
write_latex_table(show_num[["Item","Title","N_survey","Median_survey","P25_s","P75_s",
                            "N_logs","Median_logs","P25_l","P75_l","U_p","HL_diff","HL_CI_low","HL_CI_high","Cliffs_delta","Unit"]],
                  "Numeric comparisons for Q7--Q9 (Mann--Whitney U; Hodges--Lehmann median difference with 95\\% bootstrap CI; Cliff's $\\delta$).",
                  "tab:6_3_numeric", TAB/"H6_3_Q7toQ9_numeric.tex", colfmt="l l r r r r r r r r r r r r r l", index=False)

# ---------------- CATEGORICAL: Q10/Q11 per category Δp ----------------
cat_rows = []
cat_effects = []
def per_cat_diff(qcode, colS, colL):
    if not colS or not colL: return
    s = S[colS]; l = L[colL]
    cats = sorted(set(s.dropna().unique()).union(set(l.dropna().unique())))
    for cat in cats:
        k1 = int((s==cat).sum()); n1 = int(s.notna().sum())
        k0 = int((l==cat).sum()); n0 = int(l.notna().sum())
        p1 = k1/n1 if n1 else np.nan
        p0 = k0/n0 if n0 else np.nan
        dp = (p1 - p0) if (p1==p1 and p0==p0) else np.nan
        lo, hi = newcombe_diff_ci(k1, n1, k0, n0) if n1 and n0 else (np.nan, np.nan)
        try:
            table = [[k1, n1-k1],[k0, n0-k0]]
            fp = fisher_exact(table, alternative="two-sided")[1]
        except Exception:
            fp = np.nan
        cat_rows.append({"Item": qcode, "Category": str(cat), "k_s":k1,"n_s":n1,"k_l":k0,"n_l":n0,
                         "p_s":p1,"p_l":p0,"dp":dp,"ci_low":lo,"ci_high":hi,"p_fisher":fp})
        cat_effects.append({"label": f"{qcode}: {cat}", "effect": dp, "lo": lo, "hi": hi})

per_cat_diff("Q10", Q10_S, Q10_L)
per_cat_diff("Q11", Q11_S, Q11_L)

cattab = pd.DataFrame(cat_rows)
show_cat = cattab.copy()
for c in ["p_s","p_l","dp","ci_low","ci_high","p_fisher"]:
    if c in show_cat.columns: show_cat[c] = show_cat[c].astype(float).round(3)
write_latex_table(show_cat[["Item","Category","k_s","n_s","k_l","n_l","p_s","p_l","dp","ci_low","ci_high","p_fisher"]],
                  "Category share differences for Q10--Q11 (Newcombe 95\\% CI for $\\Delta p$; Fisher exact p per category).",
                  "tab:6_3_cats", TAB/"H6_3_Q10Q11_categories.tex", colfmt="l l r r r r r r r r r r", index=False)

# ---------------- TASKS: Q12 per-option Δp ----------------
cb12 = codebook[codebook["question"]=="Q12"].copy()
cb12["col"] = "Q12_" + cb12["canonical"].apply(slug)
# Neem alleen opties die in beide sets bestaan
opt_cols = [c for c in cb12["col"].tolist() if (c in S.columns and c in L.columns)]
opt_labels = {("Q12_" + slug(row["canonical"])): row["canonical"] for _, row in cb12.iterrows() if ("Q12_" + slug(row["canonical"])) in opt_cols}

task_rows = []
task_effects = []
for c in opt_cols:
    s = pd.to_numeric(S[c], errors="coerce").fillna(0).astype(int)
    l = pd.to_numeric(L[c], errors="coerce").fillna(0).astype(int)
    k1, n1 = int(s.sum()), int(len(s))
    k0, n0 = int(l.sum()), int(len(l))
    p1, p0 = k1/n1 if n1 else np.nan, k0/n0 if n0 else np.nan
    dp = (p1 - p0) if (p1==p1 and p0==p0) else np.nan
    lo, hi = newcombe_diff_ci(k1, n1, k0, n0) if n1 and n0 else (np.nan, np.nan)
    task_rows.append({"Option": opt_labels[c], "col": c, "k_s":k1,"n_s":n1,"k_l":k0,"n_l":n0,"p_s":p1,"p_l":p0,"dp":dp,"ci_low":lo,"ci_high":hi})
    task_effects.append({"label": f"Q12: {opt_labels[c]}", "effect": dp, "lo": lo, "hi": hi})

tasktab = pd.DataFrame(task_rows).sort_values("dp", ascending=False)
show_task = tasktab.copy()
for c in ["p_s","p_l","dp","ci_low","ci_high"]:
    if c in show_task.columns: show_task[c] = show_task[c].astype(float).round(3)
write_latex_table(show_task[["Option","k_s","n_s","k_l","n_l","p_s","p_l","dp","ci_low","ci_high"]],
                  "Main task options (Q12): $\\Delta p$ (survey donors $-$ logs) with Newcombe 95\\% CIs.",
                  "tab:6_3_tasks_options", TAB/"H6_3_Q12_options.tex", colfmt="l r r r r r r r r r", index=False)

# Dot-plot voor Q12 opties (gesorteerd op |Δp|)
if not tasktab.empty:
    tt = tasktab.copy()
    tt["absdp"] = tt["dp"].abs()
    tt = tt.sort_values("absdp", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.45*len(tt))))
    y = np.arange(len(tt))
    ax.hlines(y, tt["ci_low"], tt["ci_high"], linewidth=2)
    ax.plot(tt["dp"], y, "o")
    ax.set_yticks(y); ax.set_yticklabels(tt["Option"])
    ax.set_xlabel("Δp (Survey donors − Logs)"); ax.grid(axis="x", linestyle=":", linewidth=0.7)
    # Annotatie met pp
    for yi, d in zip(y, tt["dp"]):
        if pd.notna(d):
            ax.text(d + (0.01 if d>=0 else -0.01), yi, f"{d*100:.1f} pp",
                    va="center", ha="left" if d>=0 else "right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG/"fig_6_3_tasks_dots.pdf", bbox_inches="tight"); plt.close(fig)

# Q10/11 Δp dot-plot
if not cattab.empty:
    cc = cattab.copy()
    cc["absdp"] = cc["dp"].abs()
    cc = cc.sort_values("absdp", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.4*len(cc))))
    y = np.arange(len(cc))
    ax.hlines(y, cc["ci_low"], cc["ci_high"], linewidth=2)
    ax.plot(cc["dp"], y, "o")
    ax.set_yticks(y); ax.set_yticklabels(cc["Item"] + ": " + cc["Category"].astype(str))
    ax.set_xlabel("Δp (Survey donors − Logs)"); ax.grid(axis="x", linestyle=":", linewidth=0.7)
    fig.tight_layout(); fig.savefig(FIG/"fig_6_3_Q10Q11_dots.pdf", bbox_inches="tight"); plt.close(fig)

# --------------- VIOLINS/ECDF (Q7-9) ---------------
def violin_ecdf_plot(xs, xl, title, unit, save_pdf):
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_axes([0.07, 0.18, 0.42, 0.70])
    data = [pd.to_numeric(xs, errors="coerce").dropna(), pd.to_numeric(xl, errors="coerce").dropna()]
    ax1.violinplot(data, showmedians=True, showextrema=False)
    ax1.set_xticks([1,2]); ax1.set_xticklabels(["Survey donors","Logs-derived"])
    ax1.set_title(title, fontsize=11); ax1.grid(axis="y", linestyle=":", linewidth=0.6)
    if unit: ax1.set_ylabel(unit)
    ax2 = fig.add_axes([0.57, 0.18, 0.38, 0.70])
    vx, vy = ecdf(xs); lx, ly = ecdf(xl)
    if len(vx): ax2.plot(vx, vy, label="Survey donors")
    if len(lx): ax2.plot(lx, ly, label="Logs-derived")
    ax2.set_title("ECDF", fontsize=10); ax2.grid(linestyle=":", linewidth=0.6); ax2.legend(loc="lower right", fontsize=8)
    fig.savefig(save_pdf, bbox_inches="tight"); plt.close(fig)

violin_ecdf_plot(S[Q7_colS], L[Q7_colL], "Q7 sessions (last 7 days)", "", FIG/"fig_6_3_Q7_violin_ecdf.pdf")
violin_ecdf_plot(S[Q8w_colS], L[Q8w_colL], "Q8 sessions/week (from per day)", "", FIG/"fig_6_3_Q8_violin_ecdf.pdf")
violin_ecdf_plot(S[Q9_colS], L[Q9_colL], "Q9 session duration", "minutes", FIG/"fig_6_3_Q9_violin_ecdf.pdf")

# --------------- FOREST (top effects) ---------------
forest_df = pd.DataFrame(num_effects + cat_effects + task_effects)
if not forest_df.empty:
    forest_df["abs"] = forest_df["effect"].abs()
    forest_df = forest_df.sort_values("abs", ascending=True)
    # kies top 25 effecten voor leesbaarheid
    if len(forest_df) > 25:
        forest_df = forest_df.iloc[-25:]
    fig, ax = plt.subplots(figsize=(10, max(3.8, 0.5*len(forest_df))))
    y = np.arange(len(forest_df))
    ax.hlines(y, forest_df["lo"], forest_df["hi"], linewidth=2)
    ax.plot(forest_df["effect"], y, "o")
    ax.set_yticks(y); ax.set_yticklabels(forest_df["label"])
    ax.set_xlabel("Effect size (HL diff or Δp)"); ax.grid(axis="x", linestyle=":", linewidth=0.7)
    fig.tight_layout(); fig.savefig(FIG/"fig_6_3_forest_allitems.pdf", bbox_inches="tight"); plt.close(fig)

# --------------- LaTeX figure wrappers ---------------
(TAB / "H6_3_fig_Q7.tex").write_text(
    "\\begin{figure}[t]\\centering\\includegraphics[width=0.9\\linewidth]{figures/fig_6_3_Q7_violin_ecdf.pdf}"
    "\\caption{Q7: sessions last 7 days — distribution (violin) and ECDF by cohort.}"
    "\\label{fig:6_3_Q7}\\end{figure}\n", encoding="utf-8")
(TAB / "H6_3_fig_Q8.tex").write_text(
    "\\begin{figure}[t]\\centering\\includegraphics[width=0.9\\linewidth]{figures/fig_6_3_Q8_violin_ecdf.pdf}"
    "\\caption{Q8: sessions per week (from per day) — distribution and ECDF by cohort.}"
    "\\label{fig:6_3_Q8}\\end{figure}\n", encoding="utf-8")
(TAB / "H6_3_fig_Q9.tex").write_text(
    "\\begin{figure}[t]\\centering\\includegraphics[width=0.9\\linewidth]{figures/fig_6_3_Q9_violin_ecdf.pdf}"
    "\\caption{Q9: session duration — distribution and ECDF by cohort.}"
    "\\label{fig:6_3_Q9}\\end{figure}\n", encoding="utf-8")
(TAB / "H6_3_fig_Q10Q11.tex").write_text(
    "\\begin{figure}[t]\\centering\\includegraphics[width=0.95\\linewidth]{figures/fig_6_3_Q10Q11_dots.pdf}"
    "\\caption{Category share differences for Q10--Q11 with Newcombe 95\\% CIs.}"
    "\\label{fig:6_3_Q10Q11}\\end{figure}\n", encoding="utf-8")
(TAB / "H6_3_fig_tasks.tex").write_text(
    "\\begin{figure}[t]\\centering\\includegraphics[width=0.95\\linewidth]{figures/fig_6_3_tasks_dots.pdf}"
    "\\caption{Main task options (Q12): $\\Delta p$ (survey donors $-$ logs) with Newcombe 95\\% CIs.}"
    "\\label{fig:6_3_tasks}\\end{figure}\n", encoding="utf-8")
(TAB / "H6_3_fig_forest.tex").write_text(
    "\\begin{figure}[t]\\centering\\includegraphics[width=0.9\\linewidth]{figures/fig_6_3_forest_allitems.pdf}"
    "\\caption{Summary forest plot (top effects): HL median differences for Q7--Q9; $\\Delta p$ for Q10--Q12 options.}"
    "\\label{fig:6_3_forest}\\end{figure}\n", encoding="utf-8")

# --------------- finish ---------------
print("Written:")
print("-", TAB/"H6_3_Q7toQ9_numeric.tex")
print("-", TAB/"H6_3_Q10Q11_categories.tex")
print("-", TAB/"H6_3_Q12_options.tex")
print("-", TAB/"H6_3_fig_Q7.tex")
print("-", TAB/"H6_3_fig_Q8.tex")
print("-", TAB/"H6_3_fig_Q9.tex")
print("-", TAB/"H6_3_fig_Q10Q11.tex")
print("-", TAB/"H6_3_fig_tasks.tex")
print("-", TAB/"H6_3_fig_forest.tex")
print("-", FIG/"fig_6_3_Q7_violin_ecdf.pdf")
print("-", FIG/"fig_6_3_Q8_violin_ecdf.pdf")
print("-", FIG/"fig_6_3_Q9_violin_ecdf.pdf")
print("-", FIG/"fig_6_3_Q10Q11_dots.pdf")
print("-", FIG/"fig_6_3_tasks_dots.pdf")
print("-", FIG/"fig_6_3_forest_allitems.pdf")

