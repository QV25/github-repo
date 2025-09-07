# -*- coding: utf-8 -*-
"""
04_sq2_convergence.py — Survey–Log convergence (Sdon vs Llogs), verbeterde visuals

INPUT  (results/derived/):
  - Sdon_clean.csv      (n=24)
  - Llogs_clean.csv     (n=24)

MAIN FIGURES (results/fig):
  - F6_4_forest_numeric.png
  - F6_5_Q10_comp100.png
  - F6_5_Q10_deltap.png
  - F6_5_Q11_comp100.png
  - F6_5_Q11_deltap.png
  - F6_6_Q12_deltap.png
  - F6_6_task_breadth_ecdf.png

APPENDIX FIGURES:
  - F6_4_ecdf_Q7_mid.png / _Q8_mid.png / _Q9_mid.png / _usage_index.png
  - F6_4_hist_Q7_mid.png / _Q8_mid.png / _Q9_mid.png / _usage_index.png
  - F6_5_Q10_bars.png, F6_5_Q11_bars.png
  - F6_6_Q12_comp100.png, F6_7_task_pattern_scatter.png

TABLES (results/tab, ook .tex):
  - T6_4_numeric.csv/.tex   (HL shift + CI, Cliff’s δ, KS, W1, p, q)
  - T6_5_q10.csv/.tex       (per category: shares, Δp + CI, p, q)
  - T6_5_q11.csv/.tex       (idem)
  - T6_6_q12.csv/.tex       (families: shares, Δp + CI, p, q)
  - T6_6_task_breadth.csv/.tex
  - T6_7_pattern.csv/.tex   (Spearman ρ, Hellinger)

Geen seaborn, één plot per figuur, geen custom kleuren.
"""
import os, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------ Paths ------------------ #
BASE = os.getcwd()
DERIVED = os.path.join(BASE, "results", "derived")
FIG_DIR = os.path.join(BASE, "results", "fig")
TAB_DIR = os.path.join(BASE, "results", "tab")
os.makedirs(DERIVED, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
warnings.filterwarnings("ignore")
RNG = np.random.default_rng(2025)

# ------------------ Mini LaTeX writer ------------------ #
def latex_escape(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}")
             .replace("&", "\\&").replace("%", "\\%")
             .replace("$", "\\$").replace("#", "\\#")
             .replace("_", "\\_").replace("{", "\\{")
             .replace("}", "\\}").replace("~", "\\textasciitilde{}")
             .replace("^", "\\textasciicircum{}"))

def write_latex_table(df: pd.DataFrame, path: str, colfmt=None, index=False):
    cols = df.columns.tolist()
    if colfmt is None: colfmt = "l" + "r"*(len(cols)-1)
    lines = [f"\\begin{{tabular}}{{{colfmt}}}", "\\toprule"]
    lines.append(" & ".join([latex_escape(str(c)) for c in cols]) + " \\\\")
    lines.append("\\midrule")
    for _,r in df.iterrows():
        row = ["" if pd.isna(v) else str(v) for v in r.tolist()]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(lines))

# ------------------ Stats helpers ------------------ #
def average_ranks(a):
    a = np.asarray(a, dtype=float)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0; rank = 1.0
    while i < n:
        j = i
        while j+1 < n and a[order[j+1]] == a[order[i]]:
            j += 1
        avg = (rank + (rank + (j - i))) / 2.0
        ranks[order[i:j+1]] = avg
        rank += (j - i + 1)
        i = j + 1
    return ranks

def mannwhitney_u_p(x, y):
    """U and two-sided p (normal approx with tie correction, continuity-corrected)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0: return np.nan, np.nan
    allv = np.concatenate([x, y])
    ranks = average_ranks(allv)
    R1 = ranks[:n1].sum()
    U1 = R1 - n1*(n1+1)/2.0
    U2 = n1*n2 - U1
    U = min(U1, U2)
    n = n1 + n2
    _, counts = np.unique(allv, return_counts=True)
    T = np.sum(counts*(counts**2 - 1.0))
    sigma2 = (n1*n2)/12.0 * (n + 1.0 - T/(n*(n-1.0)))
    if sigma2 <= 0: return U, np.nan
    mu = n1*n2/2.0
    z = (U - mu + 0.5*np.sign(mu - U)) / math.sqrt(sigma2)
    p = 2.0*(1.0 - 0.5*(1.0 + math.erf(abs(z)/math.sqrt(2.0))))
    return float(U), float(p)

def hodges_lehmann_shift(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x)==0 or len(y)==0: return np.nan
    diffs = np.subtract.outer(x, y).ravel()
    return float(np.median(diffs))

def bootstrap_ci(func, x, y, B=10000, seed=2025):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x)==0 or len(y)==0: return (np.nan, np.nan)
    vals = np.empty(B, dtype=float); n1, n2 = len(x), len(y)
    for _ in range(B):
        xb = x[rng.integers(0, n1, n1)]; yb = y[rng.integers(0, n2, n2)]
        vals[_] = func(xb, yb)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x)==0 or len(y)==0: return np.nan
    total = 0
    for xi in x:
        total += np.sum(xi > y) - np.sum(xi < y)
    return float(total / (len(x)*len(y)))

def ks_stat(x, y):
    x = np.sort(x[~np.isnan(x)]); y = np.sort(y[~np.isnan(y)])
    if len(x)==0 or len(y)==0: return np.nan
    grid = np.unique(np.concatenate([x, y]))
    Fx = np.searchsorted(x, grid, side='right')/len(x)
    Fy = np.searchsorted(y, grid, side='right')/len(y)
    return float(np.max(np.abs(Fx - Fy)))

def wasserstein1(x, y):
    x = np.sort(x[~np.isnan(x)]); y = np.sort(y[~np.isnan(y)])
    if len(x)==0 or len(y)==0: return np.nan
    qs = np.linspace(0, 1, 200)
    return float(np.mean(np.abs(np.quantile(x, qs) - np.quantile(y, qs))))

def wilson_ci(k, n):
    if n==0: return (np.nan, np.nan)
    p = k/n; z = 1.959963984540054
    den = 1 + z**2/n
    centre = (p + z**2/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))/den
    return float(max(0, centre-half)), float(min(1, centre+half))

def newcombe_diff_ci(k1,n1,k2,n2):
    l1,u1 = wilson_ci(k1,n1); l2,u2 = wilson_ci(k2,n2)
    p1 = k1/n1 if n1>0 else np.nan; p2 = k2/n2 if n2>0 else np.nan
    return float(l1 - u2), float(u1 - l2), float(p1 - p2)

def two_prop_z(k1,n1,k2,n2):
    if n1==0 or n2==0: return np.nan
    p1,p2 = k1/n1, k2/n2; p = (k1+k2)/(n1+n2)
    se = math.sqrt(p*(1-p)*(1/n1 + 1/n2))
    if se==0: return np.nan
    z = (p1-p2)/se
    return 2.0*(1.0 - 0.5*(1.0 + math.erf(abs(z)/math.sqrt(2.0))))

def cramers_v(table):
    T = np.asarray(table, dtype=float)
    n = T.sum()
    if n==0: return np.nan
    exp = T.sum(1, keepdims=True).dot(T.sum(0, keepdims=True))/n
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((T-exp)**2/exp)
    r,c = T.shape
    denom = n*(min(r-1,c-1))
    if denom<=0: return np.nan
    return float(np.sqrt(chi2/denom))

def spearman_rho(x,y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x)<3: return np.nan
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx = (rx-rx.mean())/rx.std(ddof=0); ry = (ry-ry.mean())/ry.std(ddof=0)
    return float((rx*ry).mean())

def hellinger(p,q):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p/p.sum() if p.sum()>0 else p
    q = q/q.sum() if q.sum()>0 else q
    return float((1/np.sqrt(2))*np.sqrt(np.sum((np.sqrt(p)-np.sqrt(q))**2)))

def fdr_bh(pvals, q=0.10):
    p = np.asarray(pvals, dtype=float)
    m = np.sum(np.isfinite(p))
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(p)+1)
    qvals = p*m/ranks
    q_sorted = np.minimum.accumulate(qvals[order][::-1])[::-1]
    out = np.full_like(p, np.nan, dtype=float); out[order] = q_sorted
    return out

# ------------------ Load ------------------ #
SDON = os.path.join(DERIVED, "Sdon_clean.csv")
LLOG = os.path.join(DERIVED, "Llogs_clean.csv")
if not (os.path.exists(SDON) and os.path.exists(LLOG)):
    raise FileNotFoundError("Missing Sdon_clean.csv or Llogs_clean.csv in results/derived/.")

Sdon = pd.read_csv(SDON, dtype=str, keep_default_na=False)
Llogs = pd.read_csv(LLOG, dtype=str, keep_default_na=False)

# numeric casts
for df in (Sdon, Llogs):
    for c in ["Q7_mid","Q8_mid","Q9_mid","Q11_score","task_breadth_main"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

# pooled usage_index (across Sdon+Llogs)
pooled = pd.concat([Sdon[["Q7_mid","Q8_mid","Q9_mid"]], Llogs[["Q7_mid","Q8_mid","Q9_mid"]]], ignore_index=True)
mu = pooled.mean(numeric_only=True); sd = pooled.std(ddof=0, numeric_only=True)
Sdon["usage_index"] = ((Sdon[["Q7_mid","Q8_mid","Q9_mid"]] - mu) / sd).mean(axis=1)
Llogs["usage_index"] = ((Llogs[["Q7_mid","Q8_mid","Q9_mid"]] - mu) / sd).mean(axis=1)

# ------------------ 1) Numeric contrasts ------------------ #
NUMS = [("Q7_mid","Sessions per week"),
        ("Q8_mid","Sessions per day"),
        ("Q9_mid","Minutes per session"),
        ("usage_index","Usage index (pooled z)")]

rows = []
for col,lab in NUMS:
    x = pd.to_numeric(Sdon[col], errors="coerce").dropna().values
    y = pd.to_numeric(Llogs[col], errors="coerce").dropna().values
    HL = hodges_lehmann_shift(x,y)
    l95,u95 = bootstrap_ci(hodges_lehmann_shift, x,y, B=10000, seed=2025)
    U,p_mw = mannwhitney_u_p(x,y)
    delta = cliffs_delta(x,y)
    KS = ks_stat(x,y); W1 = wasserstein1(x,y)
    rows.append({"variable":lab,"HL_shift":HL,"l95":l95,"u95":u95,
                 "p_MW":p_mw,"Cliffs_delta":delta,"KS":KS,"W1":W1})
num_df = pd.DataFrame(rows)
num_df["q_FDR"] = fdr_bh(num_df["p_MW"].values, q=0.10)
num_df.to_csv(os.path.join(TAB_DIR,"T6_4_numeric.csv"), index=False)
L = num_df.copy()
L["HL shift [95% CI]"] = L.apply(lambda r: f"{r['HL_shift']:.2f} [{r['l95']:.2f}, {r['u95']:.2f}]", axis=1)
for k in ["Cliffs_delta","KS","W1","p_MW","q_FDR"]:
    L[k] = L[k].apply(lambda v: f"{v:.3f}")
L = L[["variable","HL shift [95% CI]","Cliffs_delta","KS","W1","p_MW","q_FDR"]].rename(columns={"variable":"Variable"})
write_latex_table(L, os.path.join(TAB_DIR,"T6_4_numeric.tex"), colfmt="l r r r r r r")

# ECDF + overlay hist (appendix)
def ecdf_plot(arr1, arr2, xlabel, outfile):
    x = np.sort(arr1[~np.isnan(arr1)]); y = np.sort(arr2[~np.isnan(arr2)])
    Fx = np.arange(1,len(x)+1)/len(x) if len(x)>0 else np.array([])
    Fy = np.arange(1,len(y)+1)/len(y) if len(y)>0 else np.array([])
    plt.figure(figsize=(6,4))
    if len(x)>0: plt.step(x,Fx,where='post',label="Sdon")
    if len(y)>0: plt.step(y,Fy,where='post',label="Llogs")
    plt.xlabel(xlabel); plt.ylabel("ECDF"); plt.title(f"ECDF: {xlabel}")
    plt.legend(); plt.tight_layout(); plt.savefig(outfile, dpi=200); plt.close()

def hist_overlay(arr1, arr2, xlabel, outfile):
    plt.figure(figsize=(6,4))
    plt.hist(arr1[~np.isnan(arr1)], bins='fd', alpha=0.5, label="Sdon", density=True)
    plt.hist(arr2[~np.isnan(arr2)], bins='fd', alpha=0.5, label="Llogs", density=True)
    plt.xlabel(xlabel); plt.ylabel("Density"); plt.title(f"Histogram: {xlabel}")
    plt.legend(); plt.tight_layout(); plt.savefig(outfile, dpi=200); plt.close()

for col,lab in NUMS:
    a = pd.to_numeric(Sdon[col], errors="coerce").values
    b = pd.to_numeric(Llogs[col], errors="coerce").values
    ecdf_plot(a,b, lab, os.path.join(FIG_DIR, f"F6_4_ecdf_{col}.png"))
    hist_overlay(a,b, lab, os.path.join(FIG_DIR, f"F6_4_hist_{col}.png"))

# Main: HL forest (sorted by |HL|)
num_sorted = num_df.copy()
num_sorted["absHL"] = num_sorted["HL_shift"].abs()
num_sorted = num_sorted.sort_values("absHL", ascending=True)
yticks = np.arange(len(num_sorted))
plt.figure(figsize=(8, max(4, 0.55*len(num_sorted)+1)))
plt.axvline(0, linewidth=1)
for i, r in enumerate(num_sorted.itertuples(index=False)):
    plt.errorbar(r.HL_shift, i, xerr=[[r.HL_shift - r.l95],[r.u95 - r.HL_shift]], fmt='o', capsize=3)
plt.yticks(yticks, num_sorted["variable"].tolist())
plt.xlabel("Hodges–Lehmann shift (Sdon − Llogs)")
plt.title("Numeric contrasts (HL shift with 95% CI)")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"F6_4_forest_numeric.png"), dpi=200); plt.close()

# ------------------ 2) Q10/Q11 ------------------ #
def canonical_q10(x):
    s = str(x).strip().lower()
    if "even" in s: return "Evenings"
    if "work" in s or "study" in s: return "During work / study hours"
    if "any" in s: return "Anytime throughout the day"
    return x

def canonical_q11(x):
    s = str(x).strip().lower()
    if "multiple" in s: return "multiple_paragraphs"
    if "short" in s and "paragraph" in s: return "short_paragraph"
    if "short" in s and "sentence" in s: return "short_sentence"
    if "var" in s: return "varies"
    if s in {"multiple_paragraphs","short_paragraph","short_sentence","varies"}: return s
    return s

def prop_block(series1, series2, ordered_cats):
    s1 = pd.Series(series1).map(lambda v: v)
    s2 = pd.Series(series2).map(lambda v: v)
    counts1 = [int((s1==c).sum()) for c in ordered_cats]
    counts2 = [int((s2==c).sum()) for c in ordered_cats]
    n1, n2 = len(s1), len(s2)
    p1 = [k/n1 if n1>0 else np.nan for k in counts1]
    p2 = [k/n2 if n2>0 else np.nan for k in counts2]
    cis = [newcombe_diff_ci(counts1[i],n1,counts2[i],n2) for i in range(len(ordered_cats))]
    deltas = [c[2] for c in cis]
    CIs = [(c[0],c[1]) for c in cis]
    pvals = [two_prop_z(counts1[i],n1,counts2[i],n2) for i in range(len(ordered_cats))]
    qvals = fdr_bh(pvals, q=0.10)
    return counts1, counts2, p1, p2, deltas, CIs, pvals, qvals, n1, n2

def comp100_plot(p1, p2, cats, title, outfile):
    plt.figure(figsize=(max(7, 1.2*len(cats)), 4))
    # Sdon
    bottom = 0.0
    for i in range(len(cats)):
        plt.bar(0, p1[i], bottom=bottom, width=0.5)
        bottom += p1[i]
    # Llogs
    bottom = 0.0
    for i in range(len(cats)):
        plt.bar(1, p2[i], bottom=bottom, width=0.5)
        bottom += p2[i]
    plt.xticks([0,1], ["Sdon","Llogs"])
    plt.ylabel("Share (composition)")
    plt.title(title)
    txt = "Order: " + " → ".join(cats)
    plt.gcf().text(0.02, 0.02, txt)
    plt.tight_layout(); plt.savefig(outfile, dpi=200); plt.close()

def deltap_forest(deltas, CIs, cats, title, outfile):
    order = np.argsort(np.abs(deltas))
    deltas = [deltas[i]*100 for i in order]
    CIs = [(CIs[i][0]*100, CIs[i][1]*100) for i in order]
    cats = [cats[i] for i in order]
    idx = np.arange(len(cats))
    plt.figure(figsize=(max(7, 1.2*len(cats)), 4))
    plt.axvline(0, linewidth=1)
    for i,(d,(l,u)) in enumerate(zip(deltas, CIs)):
        plt.errorbar(d, i, xerr=[[d-l],[u-d]], fmt='o', capsize=3)
    plt.yticks(idx, cats); plt.xlabel("Δ percentage points (Sdon − Llogs)")
    plt.title(title)
    plt.tight_layout(); plt.savefig(outfile, dpi=200); plt.close()

# Q10
Sdon["Q10_c"] = Sdon.get("Q10","").map(canonical_q10)
Llogs["Q10_c"] = Llogs.get("Q10","").map(canonical_q10)
Q10_order = ["Anytime throughout the day","During work / study hours","Evenings"]
c1,c2,p1,p2,dQ10,CIQ10,pQ10,qQ10,n1_q10,n2_q10 = prop_block(Sdon["Q10_c"], Llogs["Q10_c"], Q10_order)

df_q10 = pd.DataFrame({
    "Category": Q10_order,
    "Share Sdon": p1, "Share Llogs": p2,
    "Delta p": dQ10, "l95": [c[0] for c in CIQ10], "u95":[c[1] for c in CIQ10],
    "p": pQ10, "q_FDR": qQ10
})
df_q10.to_csv(os.path.join(TAB_DIR,"T6_5_q10.csv"), index=False)
L = df_q10.copy()
L["Share Sdon"] = L["Share Sdon"].apply(lambda v: f"{v:.3f}")
L["Share Llogs"] = L["Share Llogs"].apply(lambda v: f"{v:.3f}")
L["Δp [95% CI]"] = L.apply(lambda r: f"{r['Delta p']*100:.1f} [{r['l95']*100:.1f}, {r['u95']*100:.1f}]", axis=1)
L["p"] = L["p"].apply(lambda v: f"{v:.3f}"); L["q_FDR"] = L["q_FDR"].apply(lambda v: f"{v:.3f}")
L = L[["Category","Share Sdon","Share Llogs","Δp [95% CI]","p","q_FDR"]]
write_latex_table(L, os.path.join(TAB_DIR,"T6_5_q10.tex"), colfmt="l r r r r r")

comp100_plot(p1, p2, Q10_order, "Q10: Daypart composition (100%)", os.path.join(FIG_DIR,"F6_5_Q10_comp100.png"))
deltap_forest(dQ10, CIQ10, Q10_order, "Q10: Δ percentage points (Sdon − Llogs)", os.path.join(FIG_DIR,"F6_5_Q10_deltap.png"))

# grouped (appendix)
idx = np.arange(len(Q10_order)); w=0.4
plt.figure(figsize=(7,4))
plt.bar(idx-w/2, p1, width=w, label="Sdon"); plt.bar(idx+w/2, p2, width=w, label="Llogs")
plt.xticks(idx, Q10_order, rotation=20); plt.ylabel("Share"); plt.title("Q10: Daypart shares by cohort")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"F6_5_Q10_bars.png"), dpi=200); plt.close()

# Q11
def canonical_q11_label(code):
    return {
        "short_sentence":"One short sentence",
        "short_paragraph":"A short paragraph",
        "multiple_paragraphs":"Multiple paragraphs",
        "varies":"Varies too much to say"
    }.get(code, code)

Sdon["Q11_b"] = Sdon.get("Q11_band","").map(canonical_q11)
Llogs["Q11_b"] = Llogs.get("Q11_band","").map(canonical_q11)
Q11_order = ["short_sentence","short_paragraph","multiple_paragraphs","varies"]
Q11_labels = [canonical_q11_label(c) for c in Q11_order]
c1,c2,p1,p2,dQ11,CIQ11,pQ11,qQ11,n1_q11,n2_q11 = prop_block(Sdon["Q11_b"], Llogs["Q11_b"], Q11_order)

df_q11 = pd.DataFrame({
    "Category": Q11_labels,
    "Share Sdon": p1, "Share Llogs": p2,
    "Delta p": dQ11, "l95": [c[0] for c in CIQ11], "u95":[c[1] for c in CIQ11],
    "p": pQ11, "q_FDR": qQ11
})
df_q11.to_csv(os.path.join(TAB_DIR,"T6_5_q11.csv"), index=False)
L = df_q11.copy()
L["Share Sdon"] = L["Share Sdon"].apply(lambda v: f"{v:.3f}")
L["Share Llogs"] = L["Share Llogs"].apply(lambda v: f"{v:.3f}")
L["Δp [95% CI]"] = L.apply(lambda r: f"{r['Delta p']*100:.1f} [{r['l95']*100:.1f}, {r['u95']*100:.1f}]", axis=1)
L["p"] = L["p"].apply(lambda v: f"{v:.3f}"); L["q_FDR"] = L["q_FDR"].apply(lambda v: f"{v:.3f}")
L = L[["Category","Share Sdon","Share Llogs","Δp [95% CI]","p","q_FDR"]]
write_latex_table(L, os.path.join(TAB_DIR,"T6_5_q11.tex"), colfmt="l r r r r r")

comp100_plot(p1, p2, Q11_labels, "Q11: Prompt-length composition (100%)", os.path.join(FIG_DIR,"F6_5_Q11_comp100.png"))
deltap_forest(dQ11, CIQ11, Q11_labels, "Q11: Δ percentage points (Sdon − Llogs)", os.path.join(FIG_DIR,"F6_5_Q11_deltap.png"))

idx = np.arange(len(Q11_labels)); w=0.4
plt.figure(figsize=(8,4))
plt.bar(idx-w/2, p1, width=w, label="Sdon"); plt.bar(idx+w/2, p2, width=w, label="Llogs")
plt.xticks(idx, Q11_labels, rotation=20); plt.ylabel("Share"); plt.title("Q11: Prompt-length bands by cohort")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"F6_5_Q11_bars.png"), dpi=200); plt.close()

# ------------------ 3) Q12 families ------------------ #
FAMS = [
  ("q12__writing_and_professional_communication", "Writing & communication"),
  ("q12__brainstorming_and_personal_ideas_fun",   "Brainstorming / fun"),
  ("q12__coding_programming_help",                "Coding / programming"),
  ("q12__language_practice_or_translation",       "Language / translation"),
  ("q12__study_revision_or_exam_prep",            "Study / exam"),
  ("q12__other",                                  "Other"),
]
present = [(c,lab) for c,lab in FAMS if c in Sdon.columns and c in Llogs.columns]
rows = []
for c,lab in present:
    s1 = pd.to_numeric(Sdon[c], errors="coerce").fillna(0).astype(int)
    s2 = pd.to_numeric(Llogs[c], errors="coerce").fillna(0).astype(int)
    k1, n1 = int(s1.sum()), len(s1)
    k2, n2 = int(s2.sum()), len(s2)
    l,u,dp = newcombe_diff_ci(k1,n1,k2,n2); p = two_prop_z(k1,n1,k2,n2)
    rows.append({"family":lab,"p_Sdon":k1/n1,"p_Llogs":k2/n2,"deltap":dp,"l95":l,"u95":u,"p":p})
df_q12 = pd.DataFrame(rows)
df_q12["q_FDR"] = fdr_bh(df_q12["p"].values, q=0.10)
df_q12.to_csv(os.path.join(TAB_DIR,"T6_6_q12.csv"), index=False)

L = df_q12.copy()
L["Share Sdon"] = L["p_Sdon"].apply(lambda v: f"{v:.3f}")
L["Share Llogs"] = L["p_Llogs"].apply(lambda v: f"{v:.3f}")
L["Δp [95% CI]"] = L.apply(lambda r: f"{r['deltap']*100:.1f} [{r['l95']*100:.1f}, {r['u95']*100:.1f}]", axis=1)
L["p"] = L["p"].apply(lambda v: f"{v:.3f}")
L["q_FDR"] = L["q_FDR"].apply(lambda v: f"{v:.3f}")
L = L[["family","Share Sdon","Share Llogs","Δp [95% CI]","p","q_FDR"]].rename(columns={"family":"Q12 family"})
write_latex_table(L, os.path.join(TAB_DIR,"T6_6_q12.tex"), colfmt="l r r r r r")

# Δp forest (sorted by |Δp|)
order = np.argsort(np.abs(df_q12["deltap"].values))
labs_sorted = df_q12["family"].values[order].tolist()
dps = (df_q12["deltap"].values[order]*100).tolist()
cis = list(zip(df_q12["l95"].values[order]*100, df_q12["u95"].values[order]*100))
idx = np.arange(len(labs_sorted))
plt.figure(figsize=(8, 0.5*max(6,len(labs_sorted))+1))
plt.axvline(0, linewidth=1)
for i,(d,(l,u)) in enumerate(zip(dps, cis)):
    plt.errorbar(d, i, xerr=[[d-l],[u-d]], fmt='o', capsize=3)
plt.yticks(idx, labs_sorted); plt.xlabel("Δ percentage points (Sdon − Llogs)")
plt.title("Q12 families: Δp with 95% CI")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"F6_6_Q12_deltap.png"), dpi=200); plt.close()

# Composition 100% (appendix) — keep original order matching p arrays
labels_comp = df_q12["family"].tolist()
p1 = df_q12["p_Sdon"].values
p2 = df_q12["p_Llogs"].values
comp100_plot(p1, p2, labels_comp, "Q12 families: composition (100%)", os.path.join(FIG_DIR,"F6_6_Q12_comp100.png"))

# Task breadth contrast + ECDF
if "task_breadth_main" in Sdon.columns and "task_breadth_main" in Llogs.columns:
    xb = pd.to_numeric(Sdon["task_breadth_main"], errors="coerce").dropna().values
    yb = pd.to_numeric(Llogs["task_breadth_main"], errors="coerce").dropna().values
    HL = hodges_lehmann_shift(xb, yb); l95,u95 = bootstrap_ci(hodges_lehmann_shift, xb,yb, 10000, 2025)
    U,p_mw = mannwhitney_u_p(xb,yb); delta = cliffs_delta(xb,yb)
    pd.DataFrame([{"metric":"task_breadth_main","HL_shift":HL,"l95":l95,"u95":u95,"p_MW":p_mw,"Cliffs_delta":delta}
                 ]).to_csv(os.path.join(TAB_DIR,"T6_6_task_breadth.csv"), index=False)
    xs = np.sort(xb); ys = np.sort(yb)
    Fx = np.arange(1,len(xs)+1)/len(xs) if len(xs)>0 else np.array([])
    Fy = np.arange(1,len(ys)+1)/len(ys) if len(ys)>0 else np.array([])
    plt.figure(figsize=(6,4))
    if len(xs)>0: plt.step(xs, Fx, where='post', label="Sdon")
    if len(ys)>0: plt.step(ys, Fy, where='post', label="Llogs")
    plt.xlabel("Task breadth (Q12 families)"); plt.ylabel("ECDF")
    plt.title("Task breadth: ECDF by cohort"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,"F6_6_task_breadth_ecdf.png"), dpi=200); plt.close()

# Pattern alignment
rho = spearman_rho(df_q12["p_Sdon"].values, df_q12["p_Llogs"].values)
H = hellinger(df_q12["p_Sdon"].values, df_q12["p_Llogs"].values)
pd.DataFrame([{"Spearman_rho":rho,"Hellinger":H}]).to_csv(os.path.join(TAB_DIR,"T6_7_pattern.csv"), index=False)

# Scatter (appendix)
plt.figure(figsize=(5,5))
plt.scatter(df_q12["p_Llogs"].values, df_q12["p_Sdon"].values)
mx = float(max(df_q12["p_Llogs"].max(), df_q12["p_Sdon"].max(), 0.01))
plt.plot([0,mx],[0,mx])
for i,lab in enumerate(df_q12["family"].tolist()):
    plt.annotate(lab, (df_q12["p_Llogs"].values[i], df_q12["p_Sdon"].values[i]))
plt.xlabel("Prevalence Llogs"); plt.ylabel("Prevalence Sdon")
plt.title("Q12 prevalence alignment (Sdon vs Llogs)")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"F6_7_task_pattern_scatter.png"), dpi=200); plt.close()

print("SQ2 convergence pipeline (improved, bug-fixed) completed.")
