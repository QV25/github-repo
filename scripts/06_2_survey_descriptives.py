import pathlib, json, math
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER  = ROOT / "derived"
RES  = ROOT / "results"
TAB  = RES / "tables"
FIG  = RES / "figures"
for p in [TAB, FIG]: p.mkdir(parents=True, exist_ok=True)

# ----------------- helpers -----------------
def load_data():
    S_all  = pd.read_parquet(DER / "S_all.parquet")
    labels = json.load(open(DER / "labels_SurveyFull.json", encoding="utf-8"))
    return S_all, labels

def pick_primary_by_label(df, labels_dict, targets):
    inv = {}
    for col in df.columns:
        lab = labels_dict.get(col, "")
        inv.setdefault(str(lab).strip(), []).append(col)
    out = {}
    for tq in targets:
        hit = None
        for lab, cols in inv.items():
            if lab.startswith(tq):
                hit = cols[0]; break
        out[tq] = hit
    return out

def col_by_label_prefix(df, labels, prefix):
    for col, lab in labels.items():
        if col in df.columns and str(lab).startswith(prefix):
            return col
    return None

def wilson_ci(k, n, alpha=0.05):
    if n == 0: return (np.nan, np.nan)
    from math import sqrt
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2)
    p = k / n
    den = 1 + z*z/n
    center = p + z*z/(2*n)
    pm = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    lower = (center - pm) / den
    upper = (center + pm) / den
    return (max(0.0, lower), min(1.0, upper))

def freq_table(series, item_code, question_text, include_missing=False):
    s = series.copy()
    n_valid = s.notna().sum()
    counts = s.value_counts(dropna=True).sort_index()
    rows = []
    for cat, k in counts.items():
        p = k / n_valid if n_valid else np.nan
        lo, hi = wilson_ci(k, n_valid) if n_valid else (np.nan, np.nan)
        rows.append({
            "item": item_code,
            "question": question_text,
            "category": str(cat),
            "n": int(k), "N_valid": int(n_valid),
            "pct": float(p) if not np.isnan(p) else np.nan,
            "ci_low": float(lo) if not np.isnan(lo) else np.nan,
            "ci_high": float(hi) if not np.isnan(hi) else np.nan,
        })
    if include_missing:
        kmiss = s.isna().sum()
        if kmiss:
            rows.append({
                "item": item_code, "question": question_text,
                "category": "Missing", "n": int(kmiss), "N_valid": int(n_valid),
                "pct": np.nan, "ci_low": np.nan, "ci_high": np.nan
            })
    return rows

def numeric_summary(series, item_code, question_text):
    x = pd.to_numeric(series, errors="coerce")
    x = x.dropna()
    if len(x)==0:
        return {"item":item_code,"question":question_text,"N":0,"median":np.nan,"p25":np.nan,"p75":np.nan,"mean":np.nan,"sd":np.nan}
    return {
        "item": item_code, "question": question_text, "N": int(len(x)),
        "median": float(np.median(x)), "p25": float(np.percentile(x,25)),
        "p75": float(np.percentile(x,75)), "mean": float(x.mean()), "sd": float(x.std(ddof=1))
    }

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

def draw_bar_with_ci(ax, cats, pcts, lows, highs, title):
    x = np.arange(len(cats))
    ax.bar(x, pcts)
    # errorbars
    yerr = np.vstack([np.array(pcts)-np.array(lows), np.array(highs)-np.array(pcts)])
    ax.errorbar(x, pcts, yerr=yerr, fmt="none", capsize=3, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylim(0, max(0.01, max(pcts+[0]))*1.15)
    ax.set_ylabel("Share")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    if ct.size == 0: return np.nan
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    if n == 0: return np.nan
    r, k = ct.shape
    return math.sqrt(chi2 / (n * (min(k-1, r-1) if min(k-1, r-1)>0 else 1)))

def correlation_ratio(categories, values):
    # eta for numeric (values) grouped by categorical (categories)
    df = pd.DataFrame({"c": categories, "v": pd.to_numeric(values, errors="coerce")}).dropna()
    if df.empty: return np.nan
    groups = df.groupby("c")["v"]
    n_total = len(df)
    mu_total = df["v"].mean()
    ss_between = sum(len(g)*((g.mean()-mu_total)**2) for _, g in groups)
    ss_total   = sum((df["v"] - mu_total)**2)
    if ss_total == 0: return 0.0
    return math.sqrt(ss_between / ss_total)

# ----------------- main -----------------
S, labels = load_data()
prim = pick_primary_by_label(S, labels, [f"Q{i}" for i in range(1,21)])

# --- define groups of items ---
profile_items = [("Q1","Gender"), ("Q2","Age group"), ("Q3","Plan"), ("Q4","Device"), ("Q5","Status"), ("Q6","Field")]
usage_cat_items = [("Q10","When do you most often use ChatGPT?"),
                   ("Q11","How long are your typical prompts?")]
usage_num_items = [("Q7","sessions_7d"), ("Q8","sessions_per_day (→ per week)"), ("Q9","session_duration")]
attitude_items = [("Q19","Importance"), ("Q20","Paid-only use?")]

# 1) ---------- BIG frequency table with Wilson CIs ----------
all_rows = []

# profile + usage_cat + attitudes (categorical)
for q, _ in profile_items + usage_cat_items + attitude_items:
    col = prim.get(q)
    if not col or col not in S.columns: continue
    all_rows += freq_table(S[col], q, col, include_missing=True)

# tasks (Q12–Q17) — use dummies created earlier: r'^Q1[2-7]__'
task_dummy_cols = [c for c in S.columns if pd.Series([c]).str.match(r"^Q1[2-7]__").any()]
if task_dummy_cols:
    n_valid = len(S)  # dummies zijn 0/1 per rij
    for c in sorted(task_dummy_cols):
        k = int(pd.to_numeric(S[c], errors="coerce").fillna(0).sum())
        p = k / n_valid if n_valid else np.nan
        lo, hi = wilson_ci(k, n_valid) if n_valid else (np.nan, np.nan)
        all_rows.append({
            "item": c.split("__")[0],
            "question": c,
            "category": "Selected",
            "n": k, "N_valid": int(n_valid),
            "pct": float(p) if not np.isnan(p) else np.nan,
            "ci_low": float(lo) if not np.isnan(lo) else np.nan,
            "ci_high": float(hi) if not np.isnan(hi) else np.nan,
        })

alltab = pd.DataFrame(all_rows)
alltab_path_csv = TAB / "table_6_2_allitems.csv"
alltab.to_csv(alltab_path_csv, index=False)

# 2) ---------- Numeric summaries for Q7–Q9 (using *_mid and Q8w_mid) ----------
num_rows = []
# Q7_mid, Q8w_mid, Q9_mid created in loader
num_map = {
    "Q7": ("Q7_mid", "How many separate ChatGPT sessions did you have in the last 7 days?"),
    "Q8": ("Q8w_mid","On a typical day, how many ChatGPT sessions do you start? (→ per week)"),
    "Q9": ("Q9_mid", "On average, how long does a single ChatGPT session last?"),
}
for q, (colmid, desc) in num_map.items():
    if colmid in S.columns:
        num_rows.append(numeric_summary(S[colmid], q, desc))
numtab = pd.DataFrame(num_rows)
numtab_path_csv = TAB / "table_6_2_usage_numeric.csv"
numtab.to_csv(numtab_path_csv, index=False)

# 3) ---------- LaTeX tables ----------
# (a) all items (show selected columns)
show = alltab.copy()
show["pct"] = (show["pct"]*100).round(1)
show["ci_low"]  = (show["ci_low"]*100).round(1)
show["ci_high"] = (show["ci_high"]*100).round(1)
show.rename(columns={"item":"Item","question":"Variable","category":"Category",
                     "n":"n","N_valid":"N","pct":"%","ci_low":"CI low","ci_high":"CI high"}, inplace=True)
H6_2_all_tex = TAB / "H6_2_allitems.tex"
write_latex_table(show[["Item","Variable","Category","n","N","%","CI low","CI high"]],
                  caption="Survey descriptives with Wilson 95\\% confidence intervals.",
                  label="tab:6_2_allitems", path=H6_2_all_tex, colfmt="l l l r r r r r", index=False)

# (b) numeric usage table
numshow = numtab.copy()
for c in ["median","p25","p75","mean","sd"]:
    if c in numshow.columns: numshow[c] = numshow[c].round(2)
numshow.rename(columns={"item":"Item","question":"Variable","N":"N","median":"Median","p25":"P25",
                        "p75":"P75","mean":"Mean","sd":"SD"}, inplace=True)
H6_2_num_tex = TAB / "H6_2_usage_numeric.tex"
write_latex_table(numshow[["Item","Variable","N","Median","P25","P75","Mean","SD"]],
                  caption="Numeric summaries for usage items (midpoint coding).",
                  label="tab:6_2_usage_numeric", path=H6_2_num_tex, colfmt="l l r r r r r r", index=False)

# Also create tiny figure-include tex helpers
(TAB / "H6_2_fig_profile.tex").write_text(
    "\\begin{figure}[t]\\centering\\includegraphics[width=0.95\\linewidth]{figures/fig_6_2_profile.pdf}"
    "\\caption{Profile items (Q1--Q6): category shares with Wilson 95\\% CIs.}\\label{fig:6_2_profile}\\end{figure}\n",
    encoding="utf-8"
)
(TAB / "H6_2_fig_usage.tex").write_text(
    "\\begin{figure}[t]\\centering\\includegraphics[width=0.95\\linewidth]{figures/fig_6_2_usage.pdf}"
    "\\caption{Usage items (Q7--Q11): histograms for Q7--Q9 midpoints and category shares for Q10--Q11.}"
    "\\label{fig:6_2_usage}\\end{figure}\n",
    encoding="utf-8"
)
(TAB / "H6_2_fig_tasks.tex").write_text(
    "\\begin{figure}[t]\\centering\\includegraphics[width=0.95\\linewidth]{figures/fig_6_2_tasks.pdf}"
    "\\caption{Tasks (Q12--Q17): share of respondents selecting each task (Wilson 95\\% CIs).}"
    "\\label{fig:6_2_tasks}\\end{figure}\n",
    encoding="utf-8"
)
(TAB / "H6_2_fig_attitudes.tex").write_text(
    "\\begin{figure}[t]\\centering\\includegraphics[width=0.8\\linewidth]{figures/fig_6_2_attitudes.pdf}"
    "\\caption{Attitudes (Q19--Q20): category shares with Wilson 95\\% CIs.}"
    "\\label{fig:6_2_attitudes}\\end{figure}\n",
    encoding="utf-8"
)
(TAB / "H6_2_fig_assoc.tex").write_text(
    "\\begin{figure}[t]\\centering\\includegraphics[width=0.95\\linewidth]{figures/fig_6_2_assoc_map.pdf}"
    "\\caption{Association map: Spearman (numeric--numeric), Cram\\'er\\'s $V$ (categorical--categorical), "
    "and correlation ratio $\\eta$ (numeric--categorical).}\\label{fig:6_2_assoc}\\end{figure}\n",
    encoding="utf-8"
)

# 4) ---------- Figures ----------
# (A) Profile (Q1–Q6)
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
axes = axes.flatten()
for i, (q, _) in enumerate(profile_items):
    ax = axes[i]
    col = prim.get(q)
    if not col or col not in S.columns:
        ax.axis("off"); continue
    rows = freq_table(S[col], q, col, include_missing=False)
    ft = pd.DataFrame(rows)
    cats = ft["category"].tolist()
    pcts = ft["pct"].tolist()
    lows = ft["ci_low"].tolist()
    highs= ft["ci_high"].tolist()
    draw_bar_with_ci(ax, cats, pcts, lows, highs, f"{q}")
for j in range(len(profile_items), len(axes)): axes[j].axis("off")
fig.tight_layout()
fig.savefig(FIG/"fig_6_2_profile.pdf", bbox_inches="tight")
plt.close(fig)

# (B) Usage (Q7–Q11): 2 rows x 3 cols (3 hists + 2 bars)
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
axes = axes.flatten()
# Q7/Q8/Q9 hists from midpoints
for idx, q in enumerate(["Q7","Q8","Q9"]):
    colmid = {"Q7":"Q7_mid","Q8":"Q8w_mid","Q9":"Q9_mid"}[q]
    ax = axes[idx]
    if colmid in S.columns:
        x = pd.to_numeric(S[colmid], errors="coerce").dropna()
        ax.hist(x, bins=min(15, max(5, int(np.sqrt(len(x))+2))), edgecolor="black")
        ax.set_title(f"{q} (midpoint)", fontsize=10); ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    else:
        ax.axis("off")
# Q10/Q11 bars
for idx2, q in enumerate(["Q10","Q11"], start=3):
    ax = axes[idx2]
    col = prim.get(q)
    if not col or col not in S.columns: ax.axis("off"); continue
    ft = pd.DataFrame(freq_table(S[col], q, col, include_missing=False))
    draw_bar_with_ci(ax, ft["category"].tolist(), ft["pct"].tolist(),
                        ft["ci_low"].tolist(), ft["ci_high"].tolist(), f"{q}")
# last panel off
axes[-1].axis("off")
fig.tight_layout()
fig.savefig(FIG/"fig_6_2_usage.pdf", bbox_inches="tight")
plt.close(fig)

# (C) Tasks (Q12–Q17) dot plot
task_rows = []
if task_dummy_cols:
    n_valid = len(S)
    for c in sorted(task_dummy_cols):
        k = int(pd.to_numeric(S[c], errors="coerce").fillna(0).sum())
        p = k / n_valid if n_valid else np.nan
        lo, hi = wilson_ci(k, n_valid) if n_valid else (np.nan, np.nan)
        task_rows.append({"task": c, "p": p, "lo": lo, "hi": hi})
    td = pd.DataFrame(task_rows).sort_values("p", ascending=False)
    top = td  # alles tonen (verwacht ~6), zoniet top 20
    y = np.arange(len(top))[::-1]
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.45*len(top))))
    ax.hlines(y, top["lo"], top["hi"], linewidth=2)
    ax.plot(top["p"], y, "o")
    ax.set_yticks(y); ax.set_yticklabels(top["task"])
    ax.set_xlabel("Share"); ax.set_xlim(0, min(1.0, (top["hi"].max() if len(top) else 1.0)*1.05))
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.8)
    ax.set_title("Q12–Q17 tasks", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG/"fig_6_2_tasks.pdf", bbox_inches="tight")
    plt.close(fig)
else:
    # empty placeholder
    fig, ax = plt.subplots(figsize=(6,3)); ax.axis("off")
    fig.savefig(FIG/"fig_6_2_tasks.pdf", bbox_inches="tight"); plt.close(fig)

# (D) Attitudes (Q19–Q20) bars
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for i, q in enumerate(["Q19","Q20"]):
    ax = axes[i]
    col = prim.get(q)
    if not col or col not in S.columns: ax.axis("off"); continue
    ft = pd.DataFrame(freq_table(S[col], q, col, include_missing=False))
    draw_bar_with_ci(ax, ft["category"].tolist(), ft["pct"].tolist(),
                        ft["ci_low"].tolist(), ft["ci_high"].tolist(), f"{q}")
fig.tight_layout()
fig.savefig(FIG/"fig_6_2_attitudes.pdf", bbox_inches="tight")
plt.close(fig)

# (E) Association map
# numeric vars
num_vars = [v for v in ["usage_index","Q7_mid","Q8w_mid","Q9_mid","num_tasks"] if v in S.columns]
# categorical vars (plan, status, Q19, Q20)
cat_vars = []
for prefix in ["Q3","Q5","Q19","Q20"]:
    c = col_by_label_prefix(S, labels, prefix)
    if c: cat_vars.append(c)

vars_all = num_vars + cat_vars
K = len(vars_all)
M = np.full((K, K), np.nan, dtype=float)

for i in range(K):
    for j in range(K):
        vi, vj = vars_all[i], vars_all[j]
        if i == j:
            M[i, j] = 1.0
            continue
        # numeric-numeric -> Spearman |rho|
        if vi in num_vars and vj in num_vars:
            xi = pd.to_numeric(S[vi], errors="coerce")
            xj = pd.to_numeric(S[vj], errors="coerce")
            m = pd.concat([xi, xj], axis=1).dropna()
            if len(m) >= 3:
                rho, _ = spearmanr(m.iloc[:,0], m.iloc[:,1])
                M[i, j] = abs(rho) if rho==rho else np.nan
        # cat-cat -> Cramér's V
        elif vi in cat_vars and vj in cat_vars:
            M[i, j] = cramers_v(S[vi], S[vj])
        # mixed -> correlation ratio (eta)
        else:
            # ensure numeric is last arg in our function signature
            if vi in num_vars and vj in cat_vars:
                M[i, j] = correlation_ratio(S[vj], S[vi])
            elif vi in cat_vars and vj in num_vars:
                M[i, j] = correlation_ratio(S[vi], S[vj])

# plot heatmap
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(M, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(np.arange(K)); ax.set_yticks(np.arange(K))
ax.set_xticklabels(vars_all, rotation=45, ha="right")
ax.set_yticklabels(vars_all)
ax.set_title("Association map (|rho|, Cramér's V, eta)", fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Association (0–1)")
fig.tight_layout()
fig.savefig(FIG/"fig_6_2_assoc_map.pdf", bbox_inches="tight")
plt.close(fig)

# ----------------- finish -----------------
print("Written:")
print("-", H6_2_all_tex)
print("-", H6_2_num_tex)
print("-", TAB/"H6_2_fig_profile.tex")
print("-", TAB/"H6_2_fig_usage.tex")
print("-", TAB/"H6_2_fig_tasks.tex")
print("-", TAB/"H6_2_fig_attitudes.tex")
print("-", TAB/"H6_2_fig_assoc.tex")
print("-", FIG/"fig_6_2_profile.pdf")
print("-", FIG/"fig_6_2_usage.pdf")
print("-", FIG/"fig_6_2_tasks.pdf")
print("-", FIG/"fig_6_2_attitudes.pdf")
print("-", FIG/"fig_6_2_assoc_map.pdf")

