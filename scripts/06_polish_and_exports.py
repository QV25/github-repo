# -*- coding: utf-8 -*-
"""
06_polish_and_exports.py
Voegt 'finishing touches' toe voor H6-figuren en exporteert aanvullende tabellen:
- n-annotatie in alle main-figuren
- consistente units op assen/titels
- FDR-BH q<0.10 markering (†) in labels
- vaste categorie-orde voor Q10/Q11
- Q11 linear-by-linear trend (coef + 95% CI) + CSV-export
- Q12 'Other' als residual label
- Task-pattern scatter met diagonaal (y=x) en annotatie (Spearman ρ + 95% CI; Hellinger)
- Top-gaps labels met absolute effect rechts
- Extra statistieken: KS, Wasserstein-1, Cramér's V (met n, df), q_FDR per blok
- Subtasks top-3 Δp per familie (indicatief; met support)

Outputs: figuren in results/fig/, tabellen in results/tab/
"""

import os, re, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- Config ----------------------------- #
BASE = os.getcwd()
DERIVED = os.path.join(BASE, "results", "derived")
FIG_DIR = os.path.join(BASE, "results", "fig")
TAB_DIR = os.path.join(BASE, "results", "tab")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

RNG = np.random.default_rng(2025)
OVERWRITE = True  # False -> schrijft *_v2.png i.p.v. te overschrijven

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

warnings.filterwarnings("ignore")

# ----------------------------- Helpers ----------------------------- #
def slugify(s: str) -> str:
    s = str(s).replace(":", " ")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def panel_out(path: str) -> str:
    if OVERWRITE: return path
    root, ext = os.path.splitext(path)
    return f"{root}_v2{ext}"

def wilson_ci(k, n):
    if n == 0: return (np.nan, np.nan, np.nan)
    z = 1.959963984540054
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / den
    half = z * np.sqrt((p*(1-p))/n + z**2/(4*n**2)) / den
    return float(p), float(max(0, centre-half)), float(min(1, centre+half))

def newcombe_diff_ci(k1,n1,k2,n2):
    _,l1,u1 = wilson_ci(k1,n1)
    _,l2,u2 = wilson_ci(k2,n2)
    p1 = k1/n1 if n1>0 else np.nan
    p2 = k2/n2 if n2>0 else np.nan
    return float(p1-p2), float(l1-u2), float(u1-l2)

def two_prop_z(k1,n1,k2,n2):
    if n1==0 or n2==0: return np.nan
    p1, p2 = k1/n1, k2/n2
    p = (k1+k2)/(n1+n2)
    se = math.sqrt(p*(1-p)*(1/n1 + 1/n2))
    if se == 0: return np.nan
    z = (p1-p2)/se
    return 2.0*(1.0 - 0.5*(1.0 + math.erf(abs(z)/math.sqrt(2))))

def fdr_bh(pvals, q=0.10):
    p = np.asarray(pvals, dtype=float)
    m = np.sum(np.isfinite(p))
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(p)+1)
    qvals = p * m / ranks
    q_sorted = np.minimum.accumulate(qvals[order][::-1])[::-1]
    out = np.full_like(p, np.nan, dtype=float); out[order] = q_sorted
    return out

def ks_statistic(x, y):
    x = np.sort(np.asarray(x, dtype=float)); y = np.sort(np.asarray(y, dtype=float))
    n, m = len(x), len(y)
    if n==0 or m==0: return np.nan
    i=j=0; d=0.0
    while i<n and j<m:
        if x[i] <= y[j]:
            i += 1
        else:
            j += 1
        Fx = i/n; Fy = j/m
        d = max(d, abs(Fx-Fy))
    return float(d)

def wasserstein1(x, y):
    x = np.sort(np.asarray(x, dtype=float)); y = np.sort(np.asarray(y, dtype=float))
    n, m = len(x), len(y)
    if n==0 or m==0: return np.nan
    # union of breakpoints
    pts = np.unique(np.concatenate([x, y]))
    if len(pts) == 1: return 0.0
    i = j = 0; Fx = Fy = 0.0; w1 = 0.0
    for t_next in pts[1:]:
        t = t_next  # step at each breakpoint
        while i < n and x[i] <= t:
            i += 1
        while j < m and y[j] <= t:
            j += 1
        Fx = i/n; Fy = j/m
    # approximate integral via Riemann sum of edges between points
    # More stable approach:
    pts_full = np.concatenate([[-np.inf], pts, [np.inf]])
    i=j=0; Fx=Fy=0.0; w1=0.0
    for k in range(1, len(pts_full)-1):
        left, right = pts_full[k], pts_full[k+1]
        if not np.isfinite(left) or not np.isfinite(right): continue
        while i<n and x[i]<=left: i+=1
        while j<m and y[j]<=left: j+=1
        Fx = i/n; Fy = j/m
        w1 += abs(Fx-Fy) * (right-left)
    return float(w1)

def cramers_v(table_counts):
    T = np.asarray(table_counts, dtype=float)
    n = T.sum()
    if n == 0: return np.nan
    exp = T.sum(1, keepdims=True).dot(T.sum(0, keepdims=True))/n
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((T-exp)**2/exp)
    r, c = T.shape
    denom = n*(min(r-1, c-1))
    if denom <= 0: return np.nan
    return float(np.sqrt(chi2/denom)), int((r-1)*(c-1)), int(n)

def mannwhitney_u_p(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1==0 or n2==0: return np.nan
    allv = np.concatenate([x, y])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv)+1)
    R1 = ranks[:n1].sum()
    U1 = R1 - n1*(n1+1)/2.0
    U2 = n1*n2 - U1
    U = min(U1, U2)
    # continuity-corrected z under H0
    _, counts = np.unique(allv, return_counts=True)
    T = np.sum(counts*(counts**2 - 1.0))
    mu = n1*n2/2.0
    sigma2 = (n1*n2)/12.0 * ((n1+n2+1) - T/((n1+n2)*(n1+n2-1)))
    if sigma2 <= 0: return np.nan
    z = (U - mu + 0.5*np.sign(mu - U)) / math.sqrt(sigma2)
    p = 2.0*(1.0 - 0.5*(1.0 + math.erf(abs(z)/math.sqrt(2.0))))
    return float(p)

def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x)==0 or len(y)==0: return np.nan
    total = 0
    for xi in x:
        total += np.sum(xi > y) - np.sum(xi < y)
    return float(total / (len(x)*len(y)))

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
    vals = np.empty(B, dtype=float)
    n1, n2 = len(x), len(y)
    for b in range(B):
        xb = x[rng.integers(0,n1,n1)]; yb = y[rng.integers(0,n2,n2)]
        vals[b] = func(xb, yb)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

def logistic_irls(y, x):
    """
    2-parameter logit: logit(p) = b0 + b1*x
    Returns: b0, b1, se0, se1 (Wald), cov 2x2
    """
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]; x = x[mask]
    if y.size < 4:  # need some data
        return (np.nan, np.nan, np.nan, np.nan, np.full((2,2), np.nan))
    X = np.column_stack([np.ones_like(x), x])
    # init
    b = np.zeros(2, dtype=float)
    for _ in range(25):
        eta = X @ b
        p = 1.0/(1.0 + np.exp(-eta))
        W = p*(1-p)
        # avoid degenerate
        W = np.clip(W, 1e-6, None)
        z = eta + (y - p)/W
        XtW = X.T * W
        XtWX = XtW @ X
        XtWz = XtW @ z
        try:
            b_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            b_new = np.linalg.pinv(XtWX) @ XtWz
        if np.linalg.norm(b_new - b) < 1e-8:
            b = b_new; break
        b = b_new
    cov = np.linalg.pinv(XtWX)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    return b[0], b[1], se[0], se[1], cov

def spearman_rho_with_bootstrap(preval_Sdon, preval_Llogs, B=5000, seed=2025):
    """
    preval_* : dict {label: prevalence in [0,1]}
    Bootstrapt door donors te resamplen op labelniveau? Hier simuleren
    via multinomiale variance rond p (benaderend) door categorywise
    resampling uit Dirichlet-multinomial op basis van p en totale n.
    Eenvoudiger en stabiel: non-parametric bootstrap over donors:
    verwacht dat je deze call met donor-level 0/1 arrays aanlevert.
    In dit script: we reconstrueren 0/1 arrays vanuit  p en n → approx.
    """
    # labels gemeenschappelijk
    labels = sorted(set(preval_Sdon.keys()) | set(preval_Llogs.keys()))
    p1 = np.array([preval_Sdon.get(k,0.0) for k in labels], float)
    p2 = np.array([preval_Llogs.get(k,0.0) for k in labels], float)

    # Spearman
    def _rho(a, b):
        ra = pd.Series(a).rank(method="average").values
        rb = pd.Series(b).rank(method="average").values
        if np.std(ra)==0 or np.std(rb)==0: return np.nan
        return float(np.corrcoef(ra, rb)[0,1])

    rho_hat = _rho(p1, p2)

    # bootstrap door categorieën (klein K=6); geeft grove CI
    rng = np.random.default_rng(seed)
    boots = []
    K = len(labels)
    if K >= 3:
        for _ in range(B):
            idx = rng.integers(0, K, K)
            boots.append(_rho(p1[idx], p2[idx]))
    if len(boots)==0:
        return rho_hat, (np.nan, np.nan)
    boots = np.array(boots, float)
    return rho_hat, (float(np.nanpercentile(boots, 2.5)),
                     float(np.nanpercentile(boots, 97.5)))

# ----------------------------- Canonical mapping ----------------------------- #
def canon_q10(x):
    s = str(x).strip().lower()
    if "work" in s or "study" in s: return "During work / study hours"
    if "even" in s: return "Evenings"
    if "any" in s: return "Anytime throughout the day"
    return "Anytime throughout the day"

def canon_q11_band(x):
    s = str(x).strip().lower()
    if "multiple" in s: return "Multiple paragraphs"
    if "short" in s and "paragraph" in s: return "A short paragraph"
    if "short" in s and "sentence" in s: return "One short sentence"
    if "var" in s: return "Varies too much to say"
    if s == "multiple_paragraphs": return "Multiple paragraphs"
    if s == "short_paragraph": return "A short paragraph"
    if s == "short_sentence": return "One short sentence"
    if s == "varies": return "Varies too much to say"
    return "Varies too much to say"

Q10_ORDER = ["During work / study hours", "Evenings", "Anytime throughout the day"]
Q11_ORDER = ["One short sentence", "A short paragraph", "Multiple paragraphs", "Varies too much to say"]

Q12_FAMS = [
    ("q12__writing_and_professional_communication", "Q12: Writing & communication"),
    ("q12__brainstorming_and_personal_ideas_fun",   "Q12: Brainstorming / fun"),
    ("q12__coding_programming_help",                "Q12: Coding / programming"),
    ("q12__language_practice_or_translation",       "Q12: Language / translation"),
    ("q12__study_revision_or_exam_prep",            "Q12: Study / exam"),
    ("q12__other",                                  "Q12: Other (residual; not combined with concrete labels)"),
]

PARENT_FOR_SUBTASK = {
    "q13__": "q12__writing_and_professional_communication",
    "q14__": "q12__brainstorming_and_personal_ideas_fun",
    "q15__": "q12__coding_programming_help",
    "q16__": "q12__language_practice_or_translation",
    "q17__": "q12__study_revision_or_exam_prep",
}

# ----------------------------- Load data ----------------------------- #
def load_clean(name):
    p = os.path.join(DERIVED, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}")
    return pd.read_csv(p, dtype=str, keep_default_na=False)

S = load_clean("S_clean.csv")
Sdon = load_clean("Sdon_clean.csv")
Llogs = load_clean("Llogs_clean.csv")

# numerics
for df in (S, Sdon, Llogs):
    for col in ["Q7_mid","Q8_mid","Q9_mid","Q11_score","task_breadth_main"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

# usage_index pooled (Sdon+Llogs)
pooled = pd.concat([Sdon[["Q7_mid","Q8_mid","Q9_mid"]], Llogs[["Q7_mid","Q8_mid","Q9_mid"]]], ignore_index=True)
mu = pooled.mean(numeric_only=True); sd = pooled.std(ddof=0, numeric_only=True)
Sdon["usage_index"] = ((Sdon[["Q7_mid","Q8_mid","Q9_mid"]] - mu) / sd).mean(axis=1)
Llogs["usage_index"] = ((Llogs[["Q7_mid","Q8_mid","Q9_mid"]] - mu) / sd).mean(axis=1)

# canonical Q10/Q11
for df in (S, Sdon, Llogs):
    if "Q10" in df.columns: df["Q10_c"] = df["Q10"].map(canon_q10)
    if "Q11_band" in df.columns: df["Q11_c"] = df["Q11_band"].map(canon_q11_band)

# ----------------------------- (a) Numeric block ----------------------------- #
NUMS = [("Q7_mid","Sessions per week"),
        ("Q8_mid","Sessions per day"),
        ("Q9_mid","Minutes per session"),
        ("usage_index","Usage index (pooled z)")]

rows=[]
for col, lab in NUMS:
    x = pd.to_numeric(Sdon[col], errors="coerce").dropna().values
    y = pd.to_numeric(Llogs[col], errors="coerce").dropna().values
    HL = hodges_lehmann_shift(x,y)
    l95,u95 = bootstrap_ci(hodges_lehmann_shift, x,y, B=10000, seed=2025)
    p_mw = mannwhitney_u_p(x,y)
    delta = cliffs_delta(x,y)
    ks = ks_statistic(x, y)
    w1 = wasserstein1(x, y)
    rows.append({"component": col, "label": lab, "effect": HL, "l95": l95, "u95": u95,
                 "p": p_mw, "cliffs_delta": delta, "ks": ks, "w1": w1,
                 "n_Sdon": len(x), "n_Llogs": len(y)})
df_num_plus = pd.DataFrame(rows)
df_num_plus["q_FDR"] = fdr_bh(df_num_plus["p"].values, q=0.10)
df_num_plus.to_csv(os.path.join(TAB_DIR, "T6_4_numeric_plus.csv"), index=False)

# Forest (overschrijf)
plt.figure(figsize=(8, 5))
labels = df_num_plus["label"].tolist()
effects = df_num_plus["effect"].values
lows = df_num_plus["l95"].values
highs = df_num_plus["u95"].values
order = np.arange(len(labels))
plt.axvline(0, linewidth=1)
for i,(d,l,u) in enumerate(zip(effects, lows, highs)):
    plt.errorbar(d, i, xerr=[[d-l],[u-d]], fmt='o', capsize=3)
plt.yticks(order, labels)
plt.xlabel("Hodges–Lehmann shift (Sdon − Llogs)")
plt.title("Numeric contrasts (HL shift with 95% CI)\n" +
          f"n={df_num_plus['n_Sdon'].min()} vs n={df_num_plus['n_Llogs'].min()}")
plt.tight_layout()
plt.savefig(panel_out(os.path.join(FIG_DIR, "F6_4_forest_numeric.png"))); plt.close()

# ----------------------------- (b) Q10 shares/Δp + V ----------------------------- #
def shares_deltap(df1, df2, col, categories):
    rows=[]
    for cat in categories:
        k1 = int((df1[col]==cat).sum()); n1 = int(df1[col].notna().sum())
        k2 = int((df2[col]==cat).sum()); n2 = int(df2[col].notna().sum())
        p, l, u = wilson_ci(k1, n1)
        p2, l2, u2 = wilson_ci(k2, n2)
        dp, dl, du = newcombe_diff_ci(k1,n1,k2,n2)
        pval = two_prop_z(k1,n1,k2,n2)
        rows.append({"category": cat, "p_Sdon": p, "l_Sdon": l, "u_Sdon": u,
                     "p_Llogs": p2, "l_Llogs": l2, "u_Llogs": u2,
                     "deltap": dp, "l95": dl, "u95": du, "p": pval,
                     "n_Sdon": n1, "n_Llogs": n2})
    out = pd.DataFrame(rows)
    out["q_FDR"] = fdr_bh(out["p"].values, q=0.10)
    return out

Q10_stats = shares_deltap(Sdon, Llogs, "Q10_c", Q10_ORDER)

# Cramér's V
tab = np.zeros((len(Q10_ORDER), 2), dtype=int)
for i,cat in enumerate(Q10_ORDER):
    tab[i,0] = int((Sdon["Q10_c"]==cat).sum())
    tab[i,1] = int((Llogs["Q10_c"]==cat).sum())
V, df_v, n_v = cramers_v(tab)
Q10_stats["cramers_V"] = V
Q10_stats["cramers_df"] = df_v
Q10_stats["cramers_n"] = n_v
Q10_stats.to_csv(os.path.join(TAB_DIR, "T6_5_q10_stats.csv"), index=False)

# Bars & Δp panels met n en †
# Bars
plt.figure(figsize=(8,5))
x = np.arange(len(Q10_ORDER)); w=0.38
p1 = [Q10_stats.loc[Q10_stats['category']==c, 'p_Sdon'].values[0] for c in Q10_ORDER]
p2 = [Q10_stats.loc[Q10_stats['category']==c, 'p_Llogs'].values[0] for c in Q10_ORDER]
plt.bar(x-w/2, p1, width=w, label="Sdon")
plt.bar(x+w/2, p2, width=w, label="Llogs")
plt.xticks(x, Q10_ORDER, rotation=20)
plt.ylabel("Share")
plt.title(f"Q10: Daypart shares by cohort\nn={Q10_stats['n_Sdon'].max()} vs n={Q10_stats['n_Llogs'].max()}")
plt.legend()
plt.tight_layout()
plt.savefig(panel_out(os.path.join(FIG_DIR, "F6_5_Q10_bars.png"))); plt.close()

# Δp forest
plt.figure(figsize=(8,4.8))
plt.axvline(0, linewidth=1)
cats = Q10_ORDER
for i,cat in enumerate(cats):
    row = Q10_stats[Q10_stats["category"]==cat].iloc[0]
    label = cat + (" †" if row["q_FDR"]<0.10 else "")
    d = row["deltap"]*100; l = row["l95"]*100; u = row["u95"]*100
    plt.errorbar(d, i, xerr=[[d-l],[u-d]], fmt='o', capsize=3)
    plt.text(u + (1 if u>=0 else -1), i, f"{d:.1f} pp", va='center', fontsize=9)
plt.yticks(np.arange(len(cats)), cats)
plt.xlabel("Δ percentage points (Sdon − Llogs)")
plt.title(f"Q10: Δ percentage points with 95% CI\nCramér's V={V:.2f}, df={df_v}, n={n_v}")
plt.tight_layout()
plt.savefig(panel_out(os.path.join(FIG_DIR, "F6_5_Q10_deltap.png"))); plt.close()

# ----------------------------- (c) Q11 shares/Δp + trend + V ----------------------------- #
Q11_stats = shares_deltap(Sdon, Llogs, "Q11_c", Q11_ORDER)

# Trend: logit Sdon~score (excl 'Varies')
score_map = {"One short sentence":1, "A short paragraph":2, "Multiple paragraphs":3}
df_tr = pd.concat([
    pd.DataFrame({"cohort":"Sdon", "score": Sdon["Q11_c"].map(score_map)}),
    pd.DataFrame({"cohort":"Llogs","score": Llogs["Q11_c"].map(score_map)}),
], ignore_index=True)
df_tr = df_tr.dropna()
y = (df_tr["cohort"]=="Sdon").astype(int).values
x = df_tr["score"].astype(float).values
b0, b1, se0, se1, _ = logistic_irls(y, x)
l95, u95 = b1 - 1.96*se1, b1 + 1.96*se1
trend = pd.DataFrame([{"coef":b1, "l95":l95, "u95":u95, "n": len(df_tr)}])
trend.to_csv(os.path.join(TAB_DIR, "T6_5_q11_trend.csv"), index=False)

# Cramér's V 4x2
tab11 = np.zeros((len(Q11_ORDER), 2), dtype=int)
for i,cat in enumerate(Q11_ORDER):
    tab11[i,0] = int((Sdon["Q11_c"]==cat).sum())
    tab11[i,1] = int((Llogs["Q11_c"]==cat).sum())
V11, df11, n11 = cramers_v(tab11)
Q11_stats["cramers_V"] = V11
Q11_stats["cramers_df"] = df11
Q11_stats["cramers_n"] = n11
Q11_stats.to_csv(os.path.join(TAB_DIR, "T6_5_q11_stats.csv"), index=False)

# Bars
plt.figure(figsize=(9,5))
x = np.arange(len(Q11_ORDER)); w=0.38
p1 = [Q11_stats.loc[Q11_stats['category']==c, 'p_Sdon'].values[0] for c in Q11_ORDER]
p2 = [Q11_stats.loc[Q11_stats['category']==c, 'p_Llogs'].values[0] for c in Q11_ORDER]
plt.bar(x-w/2, p1, width=w, label="Sdon")
plt.bar(x+w/2, p2, width=w, label="Llogs")
plt.xticks(x, Q11_ORDER, rotation=20)
plt.ylabel("Share")
plt.title(f"Q11: Prompt-length bands by cohort\nn={Q11_stats['n_Sdon'].max()} vs n={Q11_stats['n_Llogs'].max()}")
plt.legend()
plt.tight_layout()
plt.savefig(panel_out(os.path.join(FIG_DIR, "F6_5_Q11_bars.png"))); plt.close()

# Δp forest
plt.figure(figsize=(9,5))
plt.axvline(0, linewidth=1)
for i,cat in enumerate(Q11_ORDER):
    row = Q11_stats[Q11_stats["category"]==cat].iloc[0]
    label = cat + (" †" if row["q_FDR"]<0.10 else "")
    d = row["deltap"]*100; l = row["l95"]*100; u = row["u95"]*100
    plt.errorbar(d, i, xerr=[[d-l],[u-d]], fmt='o', capsize=3)
    plt.text(u + (1 if u>=0 else -1), i, f"{d:.1f} pp", va='center', fontsize=9)
plt.yticks(np.arange(len(Q11_ORDER)), Q11_ORDER)
plt.xlabel("Δ percentage points (Sdon − Llogs)")
plt.title(f"Q11: Δ percentage points with 95% CI\nLinear-by-linear trend (logit slope) = {b1:.2f} "
          f"[{l95:.2f}, {u95:.2f}]")
plt.tight_layout()
plt.savefig(panel_out(os.path.join(FIG_DIR, "F6_5_Q11_deltap.png"))); plt.close()

# ----------------------------- (d) Q12 prevalentie, Δp, breadth, pattern ----------------------------- #
# Prevalentie + Δp + q_FDR
rows=[]
for col, lab in Q12_FAMS:
    if col not in Sdon.columns or col not in Llogs.columns: continue
    s1 = pd.to_numeric(Sdon[col], errors="coerce").fillna(0).astype(int)
    s2 = pd.to_numeric(Llogs[col], errors="coerce").fillna(0).astype(int)
    k1, n1 = int(s1.sum()), len(s1)
    k2, n2 = int(s2.sum()), len(s2)
    p1, l1, u1 = wilson_ci(k1, n1)
    p2, l2, u2 = wilson_ci(k2, n2)
    dp, dl, du = newcombe_diff_ci(k1,n1,k2,n2)
    p_z = two_prop_z(k1,n1,k2,n2)
    rows.append({"family": lab, "Sdon_p": p1, "Sdon_l": l1, "Sdon_u": u1,
                 "Llogs_p": p2, "Llogs_l": l2, "Llogs_u": u2,
                 "deltap": dp, "l95": dl, "u95": du, "p": p_z})
df_q12_plus = pd.DataFrame(rows)
df_q12_plus["q_FDR"] = fdr_bh(df_q12_plus["p"].values, q=0.10)
df_q12_plus.to_csv(os.path.join(TAB_DIR, "T6_6_q12_plus.csv"), index=False)

# Task breadth (medianen + HL/MWU/δ)
if "task_breadth_main" in Sdon.columns and "task_breadth_main" in Llogs.columns:
    xb = pd.to_numeric(Sdon["task_breadth_main"], errors="coerce").dropna().values
    yb = pd.to_numeric(Llogs["task_breadth_main"], errors="coerce").dropna().values
    HL = hodges_lehmann_shift(xb,yb); l95,u95 = bootstrap_ci(hodges_lehmann_shift, xb,yb, 10000, 2025)
    p_mw = mannwhitney_u_p(xb,yb); delta = cliffs_delta(xb,yb)
    med_s = float(np.median(xb)) if len(xb)>0 else np.nan
    med_l = float(np.median(yb)) if len(yb)>0 else np.nan
    pd.DataFrame([{"component":"Task breadth (Q12 families)", "median_Sdon": med_s,
                   "median_Llogs": med_l, "HL": HL, "l95": l95, "u95": u95,
                   "p": p_mw, "cliffs_delta": delta}]) \
      .to_csv(os.path.join(TAB_DIR, "T6_6_task_breadth.csv"), index=False)

# Pattern: Spearman ρ + Hellinger
prev_Sdon = {lab: df_q12_plus.loc[df_q12_plus["family"]==lab, "Sdon_p"].values[0] for _,lab in Q12_FAMS if lab in df_q12_plus["family"].values}
prev_Llogs = {lab: df_q12_plus.loc[df_q12_plus["family"]==lab, "Llogs_p"].values[0] for _,lab in Q12_FAMS if lab in df_q12_plus["family"].values}
rho_hat, (rho_l, rho_u) = spearman_rho_with_bootstrap(prev_Sdon, prev_Llogs, B=5000, seed=2025)
# Hellinger distance
p1 = np.array([prev_Sdon.get(lab,0.0) for _,lab in Q12_FAMS])
p2 = np.array([prev_Llogs.get(lab,0.0) for _,lab in Q12_FAMS])
hellinger = float(np.sqrt(np.sum((np.sqrt(p1) - np.sqrt(p2))**2))/np.sqrt(2))
pd.DataFrame([{"spearman_rho": rho_hat, "l95": rho_l, "u95": rho_u, "hellinger": hellinger}]) \
  .to_csv(os.path.join(TAB_DIR, "T6_7_pattern_ci.csv"), index=False)

# Δp forest opnieuw (†) + waarde rechts
plt.figure(figsize=(9, 5.2))
plt.axvline(0, linewidth=1)
# sorteer aflopend op |effect|
dorder = np.argsort(-np.abs(df_q12_plus["deltap"].values))
fam_sorted = df_q12_plus["family"].values[dorder]
for i, lab in enumerate(fam_sorted):
    r = df_q12_plus[df_q12_plus["family"]==lab].iloc[0]
    d = r["deltap"]*100; l = r["l95"]*100; u = r["u95"]*100
    plt.errorbar(d, i, xerr=[[d-l],[u-d]], fmt='o', capsize=3)
    plt.text(u + (1 if u>=0 else -1), i, f"{d:.1f} pp", va='center', fontsize=9)
    if r["q_FDR"]<0.10:
        fam_sorted[i] = lab + " †"
plt.yticks(np.arange(len(fam_sorted)), fam_sorted)
plt.xlabel("Δ percentage points (Sdon − Llogs)")
plt.title("Q12 families: Δp with 95% CI\n" +
          f"ρ={rho_hat:.2f} [{rho_l:.2f}, {rho_u:.2f}], Hellinger={hellinger:.2f}\n" +
          f"n={len(Sdon)} vs n={len(Llogs)}")
plt.tight_layout()
plt.savefig(panel_out(os.path.join(FIG_DIR, "F6_6_Q12_deltap.png"))); plt.close()

# Pattern scatter met diagonaal
plt.figure(figsize=(6.5, 5.5))
x = [prev_Llogs.get(lab,0.0) for _, lab in Q12_FAMS]
y = [prev_Sdon.get(lab,0.0) for _, lab in Q12_FAMS]
plt.scatter(x, y)
maxv = max(0.001, float(np.nanmax([x,y])))
plt.plot([0,1], [0,1], linewidth=1, alpha=.5)  # diagonaal
for (col,lab), xi, yi in zip(Q12_FAMS, x, y):
    plt.annotate(lab.replace("Q12: ", ""), (xi, yi), fontsize=8, xytext=(3,3), textcoords="offset points")
plt.xlabel("Llogs prevalence"); plt.ylabel("Sdon prevalence")
plt.title(f"Task-pattern alignment (Q12)\nρ={rho_hat:.2f} [{rho_l:.2f}, {rho_u:.2f}], H={hellinger:.2f}")
plt.tight_layout()
plt.savefig(panel_out(os.path.join(FIG_DIR, "F6_7_task_pattern_scatter.png"))); plt.close()

# ----------------------------- (e) Subtasks top-3 Δp per familie (indicatief) ----------------------------- #
sub_rows=[]
for prefix, parent in PARENT_FOR_SUBTASK.items():
    # verzamel alle subtask kolommen met dit prefix die in beide datasets voorkomen
    subs = sorted([c for c in Sdon.columns if c.startswith(prefix) and c in Llogs.columns])
    if len(subs)==0: continue
    parent_lab = [lab for col,lab in Q12_FAMS if col==parent][0]
    # Gate: alleen donors met parent==1 in de betreffende dataset
    gate_Sdon = pd.to_numeric(Sdon[parent], errors="coerce").fillna(0).astype(int) if parent in Sdon.columns else pd.Series([0]*len(Sdon))
    gate_Llogs= pd.to_numeric(Llogs[parent], errors="coerce").fillna(0).astype(int) if parent in Llogs.columns else pd.Series([0]*len(Llogs))
    for sub in subs:
        s1 = pd.to_numeric(Sdon[sub], errors="coerce").fillna(0).astype(int)[gate_Sdon==1]
        s2 = pd.to_numeric(Llogs[sub],errors="coerce").fillna(0).astype(int)[gate_Llogs==1]
        k1, n1 = int(s1.sum()), int(len(s1))
        k2, n2 = int(s2.sum()), int(len(s2))
        dp, dl, du = newcombe_diff_ci(k1,n1,k2,n2)
        sub_rows.append({
            "family": parent_lab, "subtask": sub, "deltap": dp*100, "l95": dl*100, "u95": du*100,
            "support_Sdon": f"{k1}/{n1}", "support_Llogs": f"{k2}/{n2}"
        })
df_sub = pd.DataFrame(sub_rows)
top3_rows=[]
if not df_sub.empty:
    for fam in df_sub["family"].unique():
        df_f = df_sub[df_sub["family"]==fam]
        df_f = df_f.reindex(np.argsort(-np.abs(df_f["deltap"].values)))
        top3_rows.append(df_f.head(3))
    df_sub_top3 = pd.concat(top3_rows, ignore_index=True)
    df_sub_top3.to_csv(os.path.join(TAB_DIR, "T6_6_subtasks_top3.csv"), index=False)
else:
    # lege placeholder
    pd.DataFrame(columns=["family","subtask","deltap","l95","u95","support_Sdon","support_Llogs"]) \
      .to_csv(os.path.join(TAB_DIR, "T6_6_subtasks_top3.csv"), index=False)

# ----------------------------- (f) Representativiteit met Cramér's V ----------------------------- #
CAT_VARS = ["Q1","Q2","Q3","Q4","Q5","Q6","Q18","Q19","Q20"]
rows=[]
for q in CAT_VARS:
    if q not in S.columns or q not in Sdon.columns: continue
    cats = sorted(list(set(S[q].dropna().unique().tolist() + Sdon[q].dropna().unique().tolist())))
    # Δp per categorie
    for c in cats:
        k1 = int((Sdon[q]==c).sum()); n1 = int(Sdon[q].notna().sum())
        k2 = int((S[q]==c).sum());   n2 = int(S[q].notna().sum())
        dp, dl, du = newcombe_diff_ci(k1,n1,k2,n2)
        rows.append({"question": q, "category": c, "deltap": dp*100, "l95": dl*100, "u95": du*100,
                     "n_Sdon": n1, "n_S": n2})
df_repr = pd.DataFrame(rows)
df_repr.to_csv(os.path.join(TAB_DIR, "T6_A_repr.csv"), index=False)

# Cramér's V per vraag
rows=[]
for q in CAT_VARS:
    if q not in S.columns or q not in Sdon.columns: continue
    cats = sorted(list(set(S[q].dropna().unique().tolist() + Sdon[q].dropna().unique().tolist())))
    T = np.zeros((len(cats), 2), dtype=int)
    for i,c in enumerate(cats):
        T[i,0] = int((Sdon[q]==c).sum())
        T[i,1] = int((S[q]==c).sum())
    V, dfv, n_all = cramers_v(T)
    rows.append({"question": q, "cramers_V": V, "df": dfv, "n": n_all})
pd.DataFrame(rows).to_csv(os.path.join(TAB_DIR, "T6_A_repr_V.csv"), index=False)

print("06_polish_and_exports.py finished. New tables/figures written.")

