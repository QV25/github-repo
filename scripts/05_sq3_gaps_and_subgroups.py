# -*- coding: utf-8 -*-
"""
05_sq3_gaps_and_subgroups.py — Gaps (Sdon vs Llogs) & subgroups (survey-only)
Robuuste versie: schrijft ook lege tabellen/figuren zonder crashen.

Inputs (results/derived/):
  - S_clean.csv
  - Sdon_clean.csv
  - Llogs_clean.csv

Outputs (kern):
  - results/fig/F6_8_topgaps_numeric.png
  - results/fig/F6_8_topgaps_timing_prompt.png
  - results/fig/F6_8_topgaps_q12.png
  - results/fig/F6_8_task_breadth_ecdf.png
  - results/tab/T6_8_numeric.csv / .tex
  - results/tab/T6_8_q10q11.csv / .tex
  - results/tab/T6_8_q12.csv / .tex
  - results/tab/T6_8_task_breadth.csv / .tex
  - (optioneel) results/tab/T6_9_models_*.{csv,tex} + fig/F6_9_coeff_*.png
  - (optioneel) results/tab/T6_A_repr.csv / .tex + fig/F6_A_repr_sdon_vs_s.png
"""

import os, math, re, warnings
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
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
warnings.filterwarnings("ignore")
RNG = np.random.default_rng(2025)

# ------------------ Helpers ------------------ #
def slugify(s: str) -> str:
    s = str(s).replace(":", " ")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def latex_escape(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}")
             .replace("&", "\\&").replace("%", "\\%")
             .replace("$", "\\$").replace("#", "\\#")
             .replace("_", "\\_").replace("{", "\\{")
             .replace("}", "\\}").replace("~", "\\textasciitilde{}")
             .replace("^", "\\textasciicircum{}"))

def write_latex_table(df: pd.DataFrame, path: str, colfmt=None, index=False):
    # Zorg dat er altijd minstens 1 kolom is
    if df is None or df.shape[1] == 0:
        df = pd.DataFrame({"_": []})
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

# ------------------ Stats utils ------------------ #
def average_ranks(a):
    a = np.asarray(a, dtype=float); n = len(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float); i = 0; rank = 1.0
    while i < n:
        j = i
        while j+1 < n and a[order[j+1]] == a[order[i]]:
            j += 1
        avg = (rank + (rank + (j - i))) / 2.0
        ranks[order[i:j+1]] = avg
        rank += (j - i + 1); i = j + 1
    return ranks

def mannwhitney_u_p(x, y):
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
    for b in range(B):
        xb = x[rng.integers(0, n1, n1)]; yb = y[rng.integers(0, n2, n2)]
        vals[b] = func(xb, yb)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x)==0 or len(y)==0: return np.nan
    total = 0
    for xi in x:
        total += np.sum(xi > y) - np.sum(xi < y)
    return float(total / (len(x)*len(y)))

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
    # return CI bounds first (so we can *display* CI) + point diff
    return float(l1 - u2), float(u1 - l2), float(p1 - p2)

def two_prop_z(k1,n1,k2,n2):
    if n1==0 or n2==0: return np.nan
    p1,p2 = k1/n1, k2/n2; p = (k1+k2)/(n1+n2)
    se = math.sqrt(p*(1-p)*(1/n1 + 1/n2))
    if se==0: return np.nan
    z = (p1-p2)/se
    return 2.0*(1.0 - 0.5*(1.0 + math.erf(abs(z)/math.sqrt(2.0))))

def fdr_bh(pvals, q=0.10):
    p = np.asarray(pvals, dtype=float)
    m = np.sum(np.isfinite(p))
    if m == 0:
        return np.full_like(p, np.nan)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(p)+1)
    qvals = p*m/ranks
    q_sorted = np.minimum.accumulate(qvals[order][::-1])[::-1]
    out = np.full_like(p, np.nan, dtype=float); out[order] = q_sorted
    return out

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

# ------------------ Load data ------------------ #
def load_csv_relaxed(candidates):
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p, dtype=str, keep_default_na=False)
    raise FileNotFoundError("Missing expected cleaned file. Looked for:\n  - " + "\n  - ".join(candidates))

S = load_csv_relaxed([
    os.path.join(DERIVED, "S_clean.csv"),
    os.path.join(DERIVED, "S.csv"),
    os.path.join(DERIVED, "Survey_clean.csv"),
])
Sdon = load_csv_relaxed([os.path.join(DERIVED, "Sdon_clean.csv")])
Llogs = load_csv_relaxed([os.path.join(DERIVED, "Llogs_clean.csv")])

for df in (S, Sdon, Llogs):
    for c in ["Q7_mid","Q8_mid","Q9_mid","Q11_score","task_breadth_main"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

# pooled usage_index (Sdon+Llogs) en survey-only usage_index
pooled = pd.concat([Sdon[["Q7_mid","Q8_mid","Q9_mid"]], Llogs[["Q7_mid","Q8_mid","Q9_mid"]]], ignore_index=True)
mu = pooled.mean(numeric_only=True); sd = pooled.std(ddof=0, numeric_only=True)
Sdon["usage_index"] = ((Sdon[["Q7_mid","Q8_mid","Q9_mid"]] - mu) / sd).mean(axis=1)
Llogs["usage_index"] = ((Llogs[["Q7_mid","Q8_mid","Q9_mid"]] - mu) / sd).mean(axis=1)
S["usage_index_survey"] = ((S[["Q7_mid","Q8_mid","Q9_mid"]].astype(float) - S[["Q7_mid","Q8_mid","Q9_mid"]].astype(float).mean())
                           / S[["Q7_mid","Q8_mid","Q9_mid"]].astype(float).std(ddof=0)).mean(axis=1)

def canon_q10(x):
    s = str(x).strip().lower()
    if "even" in s: return "Q10: Evenings"
    if "work" in s or "study" in s: return "Q10: During work / study hours"
    if "any" in s: return "Q10: Anytime throughout the day"
    return "Q10: " + (str(x).strip() or "Anytime throughout the day")

def canon_q11(x):
    s = str(x).strip().lower()
    if "multiple" in s: return "Q11: Multiple paragraphs"
    if "short" in s and "paragraph" in s: return "Q11: A short paragraph"
    if "short" in s and "sentence" in s: return "Q11: One short sentence"
    if "var" in s: return "Q11: Varies too much to say"
    if s == "multiple_paragraphs": return "Q11: Multiple paragraphs"
    if s == "short_paragraph":     return "Q11: A short paragraph"
    if s == "short_sentence":      return "Q11: One short sentence"
    if s == "varies":              return "Q11: Varies too much to say"
    return "Q11: " + (str(x).strip() or "Varies too much to say")

for df in (S, Sdon, Llogs):
    if "Q10" in df.columns: df["Q10_c"] = df["Q10"].map(canon_q10)
    if "Q11_band" in df.columns: df["Q11_c"] = df["Q11_band"].map(canon_q11)

# ------------------ A) Top gaps ------------------ #
# Numeric
NUMS = [("Q7_mid","Q7: Sessions per week"),
        ("Q8_mid","Q8: Sessions per day"),
        ("Q9_mid","Q9: Minutes per session"),
        ("usage_index","Usage index (pooled z)")]

num_rows=[]
for col,lab in NUMS:
    x = pd.to_numeric(Sdon[col], errors="coerce").dropna().values
    y = pd.to_numeric(Llogs[col], errors="coerce").dropna().values
    HL = hodges_lehmann_shift(x,y)
    l95,u95 = bootstrap_ci(hodges_lehmann_shift, x,y, 10000, 2025)
    _,p = mannwhitney_u_p(x,y)
    delta = cliffs_delta(x,y)
    num_rows.append({"component":lab,"effect":HL,"l95":l95,"u95":u95,"p":p,"aux":delta,"family":"numeric"})
df_num = pd.DataFrame(num_rows, columns=["component","effect","l95","u95","p","aux","family"])
df_num["q_FDR"] = fdr_bh(df_num["p"].values, q=0.10)
df_num.to_csv(os.path.join(TAB_DIR,"T6_8_numeric.csv"), index=False)
L = df_num.copy()
L["HL shift [95% CI]"] = L.apply(lambda r: f"{r['effect']:.2f} [{r['l95']:.2f}, {r['u95']:.2f}]", axis=1)
L["Cliff's δ"] = L["aux"].apply(lambda v: f"{v:.3f}")
L["p"] = L["p"].apply(lambda v: f"{v:.3f}")
L["q_FDR"] = L["q_FDR"].apply(lambda v: f"{v:.3f}")
write_latex_table(L[["component","HL shift [95% CI]","Cliff's δ","p","q_FDR"]].rename(columns={"component":"Component"}),
                  os.path.join(TAB_DIR,"T6_8_numeric.tex"), colfmt="l r r r r")

# Categorical helper (Q10/Q11)
def deltap_table(s1, s2):
    cats = sorted(list(set(s1.dropna().unique().tolist() + s2.dropna().unique().tolist())))
    rows=[]
    for c in cats:
        k1,n1 = int((s1==c).sum()), int(s1.shape[0])
        k2,n2 = int((s2==c).sum()), int(s2.shape[0])
        l,u,dp = newcombe_diff_ci(k1,n1,k2,n2); p = two_prop_z(k1,n1,k2,n2)
        rows.append({"component":c,"effect":dp*100,"l95":l*100,"u95":u*100,"p":p})
    # Zorg dat kolommen er zijn, ook als rows==[]
    out = pd.DataFrame(rows, columns=["component","effect","l95","u95","p"])
    out["family"] = "cat"
    out["q_FDR"]  = fdr_bh(out["p"].values, q=0.10)
    return out

df_q10 = deltap_table(Sdon.get("Q10_c", pd.Series([], dtype=str)),
                      Llogs.get("Q10_c", pd.Series([], dtype=str)))
df_q11 = deltap_table(Sdon.get("Q11_c", pd.Series([], dtype=str)),
                      Llogs.get("Q11_c", pd.Series([], dtype=str)))
df_cat = pd.concat([df_q10, df_q11], ignore_index=True) if (df_q10.shape[1]>0 or df_q11.shape[1]>0) else pd.DataFrame(columns=["component","effect","l95","u95","p","family","q_FDR"])

df_cat.to_csv(os.path.join(TAB_DIR,"T6_8_q10q11.csv"), index=False)
if df_cat.shape[1] == 0:
    # helemaal leeg (komt praktisch niet voor)
    write_latex_table(pd.DataFrame({"Component":[], "Δp [95% CI]":[], "p":[], "q_FDR":[]}),
                      os.path.join(TAB_DIR,"T6_8_q10q11.tex"), colfmt="l r r r")
else:
    L = df_cat.copy()
    if not {"effect","l95","u95"}.issubset(L.columns):
        # Schrijf lege layout
        write_latex_table(pd.DataFrame({"Component":[], "Δp [95% CI]":[], "p":[], "q_FDR":[]}),
                          os.path.join(TAB_DIR,"T6_8_q10q11.tex"), colfmt="l r r r")
    else:
        L["Δp [95% CI]"] = L.apply(lambda r: f"{r['effect']:.1f} [{r['l95']:.1f}, {r['u95']:.1f}]", axis=1)
        L["p"] = L["p"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "")
        L["q_FDR"] = L["q_FDR"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "")
        write_latex_table(L[["component","Δp [95% CI]","p","q_FDR"]].rename(columns={"component":"Component"}),
                          os.path.join(TAB_DIR,"T6_8_q10q11.tex"), colfmt="l r r r")

# Q12 families
Q12_FAMS = [
  ("q12__writing_and_professional_communication", "Q12: Writing & communication"),
  ("q12__brainstorming_and_personal_ideas_fun",   "Q12: Brainstorming / fun"),
  ("q12__coding_programming_help",                "Q12: Coding / programming"),
  ("q12__language_practice_or_translation",       "Q12: Language / translation"),
  ("q12__study_revision_or_exam_prep",            "Q12: Study / exam"),
  ("q12__other",                                  "Q12: Other"),
]
rows=[]
for col,lab in Q12_FAMS:
    if col in Sdon.columns and col in Llogs.columns:
        s1 = pd.to_numeric(Sdon[col], errors="coerce").fillna(0).astype(int)
        s2 = pd.to_numeric(Llogs[col], errors="coerce").fillna(0).astype(int)
        k1,n1 = int(s1.sum()), len(s1); k2,n2 = int(s2.sum()), len(s2)
        l,u,dp = newcombe_diff_ci(k1,n1,k2,n2); p = two_prop_z(k1,n1,k2,n2)
        rows.append({"component":lab,"effect":dp*100,"l95":l*100,"u95":u*100,"p":p,"family":"q12"})
df_q12 = pd.DataFrame(rows, columns=["component","effect","l95","u95","p","family"])
if df_q12.shape[0] > 0:
    df_q12["q_FDR"] = fdr_bh(df_q12["p"].values, q=0.10)
else:
    df_q12["q_FDR"] = []
df_q12.to_csv(os.path.join(TAB_DIR,"T6_8_q12.csv"), index=False)
L = df_q12.copy()
if df_q12.shape[0] == 0:
    write_latex_table(pd.DataFrame({"Component":[], "Δp [95% CI]":[], "p":[], "q_FDR":[]}),
                      os.path.join(TAB_DIR,"T6_8_q12.tex"), colfmt="l r r r")
else:
    L["Δp [95% CI]"] = L.apply(lambda r: f"{r['effect']:.1f} [{r['l95']:.1f}, {r['u95']:.1f}]", axis=1)
    L["p"] = L["p"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    L["q_FDR"] = L["q_FDR"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    write_latex_table(L[["component","Δp [95% CI]","p","q_FDR"]].rename(columns={"component":"Component"}),
                      os.path.join(TAB_DIR,"T6_8_q12.tex"), colfmt="l r r r")

# Task breadth + ECDF
if "task_breadth_main" in Sdon.columns and "task_breadth_main" in Llogs.columns:
    xb = pd.to_numeric(Sdon["task_breadth_main"], errors="coerce").dropna().values
    yb = pd.to_numeric(Llogs["task_breadth_main"], errors="coerce").dropna().values
    HL = hodges_lehmann_shift(xb,yb); l95,u95 = bootstrap_ci(hodges_lehmann_shift, xb,yb, 10000, 2025)
    _,p = mannwhitney_u_p(xb,yb); delta = cliffs_delta(xb,yb)
    pd.DataFrame([{"component":"Task breadth (Q12 families)","effect":HL,"l95":l95,"u95":u95,"p":p,"aux":delta}]) \
      .to_csv(os.path.join(TAB_DIR,"T6_8_task_breadth.csv"), index=False)
    L = pd.DataFrame([{"Component":"Task breadth (Q12 families)",
                       "HL shift [95% CI]": f"{HL:.2f} [{l95:.2f}, {u95:.2f}]",
                       "Cliff's δ": f"{delta:.3f}", "p": f"{p:.3f}"}])
    write_latex_table(L, os.path.join(TAB_DIR,"T6_8_task_breadth.tex"), colfmt="l r r r", index=False)
    xs = np.sort(xb); ys = np.sort(yb)
    Fx = np.arange(1,len(xs)+1)/len(xs) if len(xs)>0 else np.array([])
    Fy = np.arange(1,len(ys)+1)/len(ys) if len(ys)>0 else np.array([])
    plt.figure(figsize=(6,4))
    if len(xs)>0: plt.step(xs, Fx, where='post', label="Sdon")
    if len(ys)>0: plt.step(ys, Fy, where='post', label="Llogs")
    plt.xlabel("Task breadth (Q12 families)"); plt.ylabel("ECDF"); plt.title("Task breadth: ECDF by cohort")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"F6_8_task_breadth_ecdf.png"), dpi=200); plt.close()

# Forest helpers
def forest(df, title, outpng, xlab):
    if df is None or df.shape[0] == 0 or not {"effect","l95","u95","component"}.issubset(df.columns):
        # Schrijf een placeholder i.p.v. crashen
        plt.figure(figsize=(7,3))
        plt.text(0.5, 0.5, f"No data for: {title}", ha='center', va='center')
        plt.axis("off")
        plt.tight_layout(); plt.savefig(outpng, dpi=200); plt.close()
        print(f"[info] Placeholder written for {outpng}")
        return
    order = np.argsort(np.abs(df["effect"].values))
    comp = df["component"].values[order].tolist()
    eff  = df["effect"].values[order].tolist()
    cis  = list(zip(df["l95"].values[order].tolist(), df["u95"].values[order].tolist()))
    idx = np.arange(len(comp))
    plt.figure(figsize=(10, max(4, 0.55*len(comp)+1)))
    plt.axvline(0, linewidth=1)
    for i,(d,(l,u)) in enumerate(zip(eff,cis)):
        plt.errorbar(d, i, xerr=[[d-l],[u-d]], fmt='o', capsize=3)
        plt.text(u + (0.5 if u>=0 else -0.5), i, f"{d:.1f} pp" if "Δ" in xlab else f"{d:.2f}", va='center', fontsize=9)
    plt.yticks(idx, comp); plt.xlabel(xlab); plt.title(title)
    plt.tight_layout(); plt.savefig(outpng, dpi=200); plt.close()

forest(df_num, "Top gaps: numeric (HL shift with 95% CI)",
       os.path.join(FIG_DIR,"F6_8_topgaps_numeric.png"),
       "Hodges–Lehmann shift (Sdon − Llogs)")
forest(df_cat, "Top gaps: timing & prompt bands (Δp with 95% CI)",
       os.path.join(FIG_DIR,"F6_8_topgaps_timing_prompt.png"),
       "Δ percentage points (Sdon − Llogs)")
forest(df_q12, "Top gaps: Q12 families (Δp with 95% CI)",
       os.path.join(FIG_DIR,"F6_8_topgaps_q12.png"),
       "Δ percentage points (Sdon − Llogs)")

# ------------------ B) Subgroup models (S survey-only; LPM/OLS HC3) ------------------ #
def canon_age(q2): return str(q2).strip() or "18–24"
def canon_plan(q3):
    s = str(q3).lower().strip()
    if "pro" in s:  return "Pro"
    if "plus" in s: return "Plus"
    return "Free"
def canon_device(q4):
    s = str(q4).lower().strip()
    if "smart" in s: return "Smartphone"
    if "mix" in s or "both" in s: return "Mixed equally"
    return "Laptop / desktop"
def canon_status(q5):
    s = str(q5).lower().strip()
    if "student" in s and "work" in s: return "Both student and working"
    if "student" in s: return "Student"
    if "work" in s or "employ" in s: return "Working"
    return "Other"
def canon_field(q6):
    s = str(q6).lower().strip()
    if "stem" in s or "engineer" in s or "tech" in s or "math" in s: return "STEM"
    if "business" in s or "econom" in s: return "Business / economics"
    if "creative" in s or "media" in s: return "Creative arts / media"
    if "human" in s or "social" in s: return "Humanities / social sciences"
    if "prefer" in s: return "Prefer not to say"
    return "Other"

def build_design(S_df):
    Sx = S_df.copy()
    Sx["Q2"] = Sx.get("Q2","").map(canon_age)
    Sx["Q3"] = Sx.get("Q3","").map(canon_plan)
    Sx["Q4"] = Sx.get("Q4","").map(canon_device)
    Sx["Q5"] = Sx.get("Q5","").map(canon_status)
    Sx["Q6"] = Sx.get("Q6","").map(canon_field)
    def dummies(series, prefix, drop):
        d = pd.get_dummies(series, prefix=prefix)
        if f"{prefix}_{drop}" in d.columns: d = d.drop(columns=[f"{prefix}_{drop}"])
        return d
    X = pd.concat([
        dummies(Sx["Q3"], "Q3", "Free"),
        dummies(Sx["Q4"], "Q4", "Laptop / desktop"),
        dummies(Sx["Q5"], "Q5", "Working"),
        dummies(Sx["Q6"], "Q6", "STEM"),
        dummies(Sx["Q2"], "Q2", "25–34"),
    ], axis=1).astype(float)
    X.insert(0, "const", 1.0)
    return X

X = build_design(S)

def ols_hc3(X: np.ndarray, y: np.ndarray):
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]; y = y[mask]
    if X.shape[0] == 0:
        return np.zeros(X.shape[1]), np.zeros(X.shape[1]), mask
    XT = X.T
    XtX = XT @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (XT @ y)
    resid = y - X @ beta
    H = X @ XtX_inv @ XT
    h = np.clip(np.diag(H), 0, 0.999999)
    w = (resid / (1 - h)) ** 2
    S = (X.T * w) @ X
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    return beta, se, mask

def standardize_Xy(X_df: pd.DataFrame, y: np.ndarray):
    X = X_df.copy()
    for c in X.columns[1:]:
        s = X[c].astype(float); sd = s.std(ddof=0)
        X[c] = (s - s.mean())/sd if sd>0 else 0.0
    y = np.asarray(y, dtype=float)
    y_sd = np.nanstd(y, ddof=0); y_mu = np.nanmean(y)
    y_std = (y - y_mu)/y_sd if y_sd>0 else y*0.0
    return X, y_std

def coef_plot(beta, se, names, title, outpng, top_k=10):
    keep = [i for i,n in enumerate(names) if n!="const"]
    beta = beta[keep]; se = se[keep]; names = [names[i] for i in keep]
    order = np.argsort(np.abs(beta))
    beta = beta[order]; se = se[order]; names = [names[i] for i in order]
    if len(beta) > top_k:
        beta = beta[-top_k:]; se = se[-top_k:]; names = names[-top_k:]
    y = np.arange(len(beta))
    plt.figure(figsize=(max(8, 0.6*len(names)+5), 4))
    plt.axvline(0, linewidth=1)
    for i,(b,s) in enumerate(zip(beta, se)):
        plt.errorbar(b, i, xerr=1.96*s, fmt='o', capsize=3)
    plt.yticks(y, names)
    plt.xlabel("Standardized beta (HC3)"); plt.title(title)
    plt.tight_layout(); plt.savefig(outpng, dpi=200); plt.close()

# Targets
numeric_targets = [
    ("Q7_mid", "Q7: Sessions per week"),
    ("Q8_mid", "Q8: Sessions per day"),
    ("Q9_mid", "Q9: Minutes per session"),
    ("usage_index_survey", "Usage index (survey)"),
]

# Top3 selectors — veilig bij lege df's
top3_q10q11 = []
if df_cat.shape[0] > 0 and "effect" in df_cat.columns:
    top3_q10q11 = df_cat.reindex(np.argsort(-np.abs(df_cat["effect"].values))).head(3)["component"].tolist()

top3_q12 = []
if df_q12.shape[0] > 0 and "effect" in df_q12.columns:
    top3_q12 = df_q12.reindex(np.argsort(-np.abs(df_q12["effect"].values))).head(3)["component"].tolist()

# Numeric outcomes (survey-only models)
X_design = build_design(S)
for col,label in numeric_targets:
    if col not in S.columns: 
        continue
    y = pd.to_numeric(S[col], errors="coerce").values
    Xstd, ystd = standardize_Xy(X_design, y)
    beta, se, _ = ols_hc3(Xstd.values, ystd)
    names = Xstd.columns.tolist()
    dfm = pd.DataFrame({"term":names, "beta":beta, "se":se})
    base = slugify(col)
    dfm.to_csv(os.path.join(TAB_DIR, f"T6_9_models_{base}.csv"), index=False)
    L = dfm.copy(); L["beta"]=L["beta"].apply(lambda v: f"{v:.3f}"); L["se"]=L["se"].apply(lambda v: f"{v:.3f}")
    write_latex_table(L.rename(columns={"term":"Term","beta":"Std. beta","se":"HC3 s.e."}),
                      os.path.join(TAB_DIR, f"T6_9_models_{base}.tex"), colfmt="l r r", index=False)
    coef_plot(beta, se, names, f"Profile associations: {label}",
              os.path.join(FIG_DIR, f"F6_9_coeff_{base}.png"))

# Binary outcomes (Q10/Q11 top3 + Q12 top3) -> LPM met HC3
def make_binary_indicator(S_df, comp_label):
    if comp_label.startswith("Q10:"):
        return (S_df.get("Q10_c","").astype(str) == comp_label).astype(float).values
    if comp_label.startswith("Q11:"):
        return (S_df.get("Q11_c","").astype(str) == comp_label).astype(float).values
    # Map label naar kolomnaam (Q12)
    q12_map = {
        "Q12: Writing & communication": "q12__writing_and_professional_communication",
        "Q12: Brainstorming / fun": "q12__brainstorming_and_personal_ideas_fun",
        "Q12: Coding / programming": "q12__coding_programming_help",
        "Q12: Language / translation": "q12__language_practice_or_translation",
        "Q12: Study / exam": "q12__study_revision_or_exam_prep",
        "Q12: Other": "q12__other",
    }
    if comp_label in q12_map and q12_map[comp_label] in S_df.columns:
        return pd.to_numeric(S_df[q12_map[comp_label]], errors="coerce").fillna(0).astype(float).values
    return np.zeros(S_df.shape[0], dtype=float)

for comp in top3_q10q11 + top3_q12:
    y = make_binary_indicator(S, comp)
    Xstd, ystd = standardize_Xy(X_design, y)
    beta, se, _ = ols_hc3(Xstd.values, ystd)
    names = Xstd.columns.tolist()
    dfm = pd.DataFrame({"term":names, "beta":beta, "se":se})
    base = slugify(comp)
    dfm.to_csv(os.path.join(TAB_DIR, f"T6_9_models_{base}.csv"), index=False)
    L = dfm.copy(); L["beta"]=L["beta"].apply(lambda v: f"{v:.3f}"); L["se"]=L["se"].apply(lambda v: f"{v:.3f}")
    write_latex_table(L.rename(columns={"term":"Term","beta":"Std. beta (LPM)","se":"HC3 s.e."}),
                      os.path.join(TAB_DIR, f"T6_9_models_{base}.tex"), colfmt="l r r", index=False)
    coef_plot(beta, se, names, f"Profile associations: {comp}",
              os.path.join(FIG_DIR, f"F6_9_coeff_{base}.png"))

# ------------------ C) Representativeness Sdon vs S ------------------ #
CAT_VARS = ["Q1","Q2","Q3","Q4","Q5","Q6","Q18","Q19","Q20"]
rows=[]
for q in CAT_VARS:
    if q not in S.columns or q not in Sdon.columns: continue
    cats = sorted(list(set(S[q].dropna().unique().tolist() + Sdon[q].dropna().unique().tolist())))
    for c in cats:
        k1 = int((Sdon[q]==c).sum()); n1 = int(Sdon.shape[0])
        k2 = int((S[q]==c).sum());   n2 = int(S.shape[0])
        l,u,dp = newcombe_diff_ci(k1,n1,k2,n2)
        rows.append({"component": f"{q}: {c}", "effect": dp*100, "l95": l*100, "u95": u*100})
df_repr = pd.DataFrame(rows, columns=["component","effect","l95","u95"])
df_repr.to_csv(os.path.join(TAB_DIR,"T6_A_repr.csv"), index=False)
L = df_repr.copy()
L["Δp [95% CI]"] = L.apply(lambda r: f"{r['effect']:.1f} [{r['l95']:.1f}, {r['u95']:.1f}]", axis=1)
write_latex_table(L[["component","Δp [95% CI]"]].rename(columns={"component":"Component"}),
                  os.path.join(TAB_DIR,"T6_A_repr.tex"), colfmt="l r", index=False)
# Plot
if df_repr.shape[0] > 0:
    order = np.argsort(np.abs(df_repr["effect"].values))
    comp = df_repr["component"].values[order].tolist()
    eff  = df_repr["effect"].values[order].tolist()
    cis  = list(zip(df_repr["l95"].values[order].tolist(), df_repr["u95"].values[order].tolist()))
    idx = np.arange(len(comp))
    plt.figure(figsize=(12, max(5, 0.45*len(comp)+1)))
    plt.axvline(0, linewidth=1)
    for i,(d,(l,u)) in enumerate(zip(eff,cis)):
        plt.errorbar(d, i, xerr=[[d-l],[u-d]], fmt='o', capsize=3)
    plt.yticks(idx, comp)
    plt.xlabel("Δ percentage points (Sdon − Survey full)")
    plt.title("Representativeness: Sdon vs S (Q1–Q6, Q18–Q20)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"F6_A_repr_sdon_vs_s.png"), dpi=200); plt.close()
else:
    plt.figure(figsize=(7,3)); plt.text(0.5,0.5,"No representativeness data",ha='center',va='center')
    plt.axis("off"); plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,"F6_A_repr_sdon_vs_s.png"), dpi=200); plt.close()

# ------------------ D) Sensitivity ------------------ #
def sensitivity_replace(arr, old_val, new_val):
    a = np.asarray(arr, dtype=float); a2 = a.copy()
    a2[np.isclose(a2, old_val, equal_nan=False)] = new_val
    return a2

def sens_entry(name, x, y, old, new):
    HL = hodges_lehmann_shift(x,y); l,u = bootstrap_ci(hodges_lehmann_shift, x,y, 10000, 2025)
    x2 = sensitivity_replace(x, old, new); y2 = sensitivity_replace(y, old, new)
    HL2 = hodges_lehmann_shift(x2,y2); l2,u2 = bootstrap_ci(hodges_lehmann_shift, x2,y2, 10000, 2025)
    return {"measure":name,"HL_orig":HL,"l95":l,"u95":u,"HL_sens":HL2,"l95_s":l2,"u95_s":u2,
            "direction_stable": np.sign(HL)==np.sign(HL2) or (HL==0 and HL2==0)}

sens_rows=[]
x = pd.to_numeric(Sdon["Q7_mid"], errors="coerce").dropna().values
y = pd.to_numeric(Llogs["Q7_mid"], errors="coerce").dropna().values
sens_rows.append(sens_entry("Q7 (12→15)", x, y, 12.0, 15.0))
x = pd.to_numeric(Sdon["Q8_mid"], errors="coerce").dropna().values
y = pd.to_numeric(Llogs["Q8_mid"], errors="coerce").dropna().values
sens_rows.append(sens_entry("Q8 (6→7)", x, y, 6.0, 7.0))
x = pd.to_numeric(Sdon["Q9_mid"], errors="coerce").dropna().values
y = pd.to_numeric(Llogs["Q9_mid"], errors="coerce").dropna().values
sens_rows.append(sens_entry("Q9 (>60: 75→90)", x, y, 75.0, 90.0))

df_sens = pd.DataFrame(sens_rows, columns=["measure","HL_orig","l95","u95","HL_sens","l95_s","u95_s","direction_stable"])
df_sens.to_csv(os.path.join(TAB_DIR,"T6_sensitivity.csv"), index=False)
L = df_sens.copy()
L["HLorig [95% CI]"] = L.apply(lambda r: f"{r['HL_orig']:.2f} [{r['l95']:.2f}, {r['u95']:.2f}]", axis=1)
L["HLsens [95% CI]"] = L.apply(lambda r: f"{r['HL_sens']:.2f} [{r['l95_s']:.2f}, {r['u95_s']:.2f}]", axis=1)
L["Stable?"] = L["direction_stable"].map(lambda v: "yes" if v else "no")
write_latex_table(L[["measure","HLorig [95% CI]","HLsens [95% CI]","Stable?"]].rename(columns={"measure":"Measure"}),
                  os.path.join(TAB_DIR,"T6_sensitivity.tex"), colfmt="l r r l", index=False)

print("SQ3 gaps & subgroups completed.")
