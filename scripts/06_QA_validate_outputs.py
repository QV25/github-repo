import pathlib, json, math, os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER  = ROOT / "derived"
RES  = ROOT / "results"
TAB  = RES / "tables"
FIG  = RES / "figures"

def ok(x): return "OK" if x else "FAIL"

# ---------- helpers ----------
def wilson_ci(k, n, alpha=0.05):
    if n == 0: return (np.nan, np.nan)
    from math import sqrt
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2); p = k/n
    den = 1 + z*z/n; center = p + z*z/(2*n)
    pm = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    return (max(0.0,(center-pm)/den), min(1.0,(center+pm)/den))

def smd_binary(x1, x0):
    p1 = np.nanmean(x1.astype(float)); p0 = np.nanmean(x0.astype(float))
    p  = (p1 + p0) / 2.0
    den = math.sqrt(max(p*(1-p), 0.0)) if p not in (0.0, 1.0) else np.nan
    return (p1 - p0) / den if isinstance(den, float) and den>0 else np.nan

def smd_cont(x1, x0):
    x1 = pd.to_numeric(x1, errors="coerce"); x0 = pd.to_numeric(x0, errors="coerce")
    n1, n0 = x1.notna().sum(), x0.notna().sum()
    mu1, mu0 = x1.mean(), x0.mean()
    s1, s0   = x1.std(ddof=1), x0.std(ddof=1)
    sp = np.sqrt(((max(n1-1,0))*s1**2 + (max(n0-1,0))*s0**2) / max(n1+n0-2,1))
    return (mu1-mu0)/sp if sp and sp>0 else np.nan

def cliff_delta(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().values
    if len(x)==0 or len(y)==0: return np.nan
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x)*len(y))

def hodges_lehmann(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().values
    if len(x)==0 or len(y)==0: return np.nan
    diffs = (x.reshape(-1,1) - y.reshape(1,-1)).ravel()
    return float(np.median(diffs))

def file_exists_nonzero(p: pathlib.Path):
    return p.exists() and p.is_file() and p.stat().st_size > 0

# ---------- load core data ----------
S_all  = pd.read_parquet(DER/"S_all.parquet")
S_don  = pd.read_parquet(DER/"S_donors.parquet")
L_logs = pd.read_parquet(DER/"L_logs.parquet")
labels = json.load(open(DER/"labels_SurveyFull.json", encoding="utf-8"))

def col_by_label(prefix):
    for col, lab in labels.items():
        if col in S_all.columns and str(lab).startswith(prefix):
            return col
    return None

# =========================================
# 6.1 — FUNNEL & BALANCE
# =========================================
print("== 6.1 Funnel & Balance ==")
# Funnel check
funnel_csv = TAB/"table_6_1_funnel.csv"
funnel_ok = False
if funnel_csv.exists():
    ftab = pd.read_csv(funnel_csv)
    expected = {
        "Survey responses": len(S_all),
        "Survey donors (subset)": int(S_all.get("is_donor", pd.Series([], dtype=bool)).sum()),
        "Logs-derived donors": len(L_logs),
    }
    match = all(int(ftab.loc[ftab["step"]==k,"n"].values[0]) == v for k,v in expected.items())
    funnel_ok = match
    print("Funnel CSV:", ok(match), f"→ expected {expected}")
else:
    print("Funnel CSV:", ok(False), "(missing)")

# Balance SMD check
bal_csv = TAB/"table_6_1_balance_smd.csv"
if bal_csv.exists():
    bal = pd.read_csv(bal_csv)
    # recompute
    vars_map = {
        "gender": col_by_label("Q1"),
        "age_band": col_by_label("Q2"),
        "plan": col_by_label("Q3"),
        "status": col_by_label("Q5"),
    }
    S = S_all.copy()
    S["is_donor"] = S["is_donor"].astype(bool)
    g1 = S[S["is_donor"]]; g0 = S[~S["is_donor"]]
    rec = []
    for var, col in vars_map.items():
        if not col:
            rec.append({"variable":var,"level":"<not found>","SMD":np.nan}); continue
        series = S[col].astype("object").fillna("Missing").astype("category")
        dummies = pd.get_dummies(series, prefix=var, drop_first=False)
        for dcol in dummies.columns:
            level = dcol.split("_",1)[1] if "_" in dcol else dcol
            s1 = dummies.loc[g1.index, dcol]
            s0 = dummies.loc[g0.index, dcol]
            rec.append({"variable":var, "level":level, "SMD": smd_binary(s1, s0)})
    rec.append({"variable":"usage_index","level":"(continuous)","SMD": smd_cont(g1["usage_index"], g0["usage_index"])})
    recdf = pd.DataFrame(rec)
    # compare
    merged = pd.merge(
        bal[["variable","level","SMD"]],
        recdf[["variable","level","SMD"]],
        on=["variable","level"], suffixes=("_file","_recomp"), how="outer"
    )
    merged["diff"] = (merged["SMD_file"] - merged["SMD_recomp"]).abs()
    n_big = int((merged["diff"] > 1e-6).sum())
    print(f"Balance SMD CSV: {ok(n_big==0)} → diffs>1e-6: {n_big}")
else:
    print("Balance SMD CSV:", ok(False), "(missing)")

# =========================================
# 6.2 — SURVEY DESCRIPTIVES
# =========================================
print("\n== 6.2 Survey descriptives ==")
allitems_csv = TAB/"table_6_2_allitems.csv"
usenum_csv   = TAB/"table_6_2_usage_numeric.csv"
if allitems_csv.exists():
    ai = pd.read_csv(allitems_csv)
    # sanity: 0<=p<=1
    bad = int(((ai["pct"]<0)|(ai["pct"]>1)).sum())
    # per non-task item: sum of pct ~ 1 (ignore 'Missing')
    def is_task_row(row): return str(row["item"]).startswith(("Q12","Q13","Q14","Q15","Q16","Q17"))
    sums = (ai[~ai["category"].eq("Missing") & ~ai.apply(is_task_row, axis=1)]
            .groupby("item")["pct"].sum().reset_index(name="sum"))
    n_off = int((sums["sum"].sub(1.0).abs() > 0.02).sum())  # allow 2% slack
    print("Allitems CSV:", ok(bad==0 and n_off==0), f"→ pct out-of-range: {bad}, items not summing≈1: {n_off}")
else:
    print("Allitems CSV:", ok(False), "(missing)")

if usenum_csv.exists():
    un = pd.read_csv(usenum_csv)
    # recompute medians/IQR
    recompute = []
    num_map = {"Q7":"Q7_mid","Q8":"Q8w_mid","Q9":"Q9_mid"}
    for q, col in num_map.items():
        if col in S_all.columns:
            x = pd.to_numeric(S_all[col], errors="coerce").dropna()
            if len(x)==0:
                med=p25=p75=mean=sd=np.nan; N=0
            else:
                med, p25, p75 = float(np.median(x)), float(np.percentile(x,25)), float(np.percentile(x,75))
                mean, sd = float(x.mean()), float(x.std(ddof=1)); N = int(len(x))
            recompute.append({"item":q,"N":N,"median":med,"p25":p25,"p75":p75,"mean":mean,"sd":sd})
    rd = pd.DataFrame(recompute)
    merged = pd.merge(un, rd, on="item", how="left", suffixes=("_file","_recomp"))
    diffs = []
    for c in ["median","p25","p75","mean","sd"]:
        diffs.append((merged[f"{c}_file"] - merged[f"{c}_recomp"]).abs().max())
    maxdiff = float(np.nanmax(diffs))
    print("Usage numeric CSV:", ok(maxdiff < 1e-6), f"→ max abs diff: {maxdiff:.3g}")
else:
    print("Usage numeric CSV:", ok(False), "(missing)")

# =========================================
# 6.3 — DONORS vs LOGS (Q7–Q17)
# =========================================
print("\n== 6.3 Donors vs Logs ==")
def safemw(x, y):
    try:
        return mannwhitneyu(pd.to_numeric(x, errors="coerce").dropna(),
                            pd.to_numeric(y, errors="coerce").dropna(),
                            alternative="two-sided").pvalue
    except Exception:
        return np.nan

for q, col in {"Q7":"Q7_mid","Q8":"Q8w_mid","Q9":"Q9_mid"}.items():
    if col in S_don.columns and col in L_logs.columns:
        xs = pd.to_numeric(S_don[col], errors="coerce")
        xl = pd.to_numeric(L_logs[col], errors="coerce")
        Ns, Nl = xs.notna().sum(), xl.notna().sum()
        med_s, med_l = xs.median(skipna=True), xl.median(skipna=True)
        hl = hodges_lehmann(xs, xl); cd = cliff_delta(xs, xl); p = safemw(xs, xl)
        print(f"{q}: Ns={Ns}, Nl={Nl}, med_s={med_s:.3g}, med_l={med_l:.3g}, HL={hl:.3g}, Cliff={cd:.3g}, U_p={p:.3g}")

# categorical Q10/Q11 quick Δp
def dp_share(s, l, cat):
    s, l = s.dropna(), l.dropna()
    k1, n1 = int((s==cat).sum()), int(len(s))
    k0, n0 = int((l==cat).sum()), int(len(l))
    p1, p0 = (k1/n1 if n1 else np.nan), (k0/n0 if n0 else np.nan)
    return (p1-p0, k1, n1, k0, n0)

for q in ["Q10","Q11"]:
    colS = col_by_label(q)
    colL = colS if colS in L_logs.columns else None
    if colS and colL:
        cats = sorted(set(S_don[colS].dropna().unique()).union(set(L_logs[colL].dropna().unique())))
        tops = []
        for c in cats:
            dp, k1, n1, k0, n0 = dp_share(S_don[colS], L_logs[colL], c)
            tops.append((abs(dp), dp, c, n1, n0))
        tops.sort(reverse=True)
        top = tops[0] if tops else None
        if top:
            print(f"{q}: top |Δp|={top[0]*100:.1f} pp → cat='{top[2]}' (Δp={top[1]*100:.1f} pp; Ns={top[3]}, Nl={top[4]})")

# tasks Q12–Q17 quick Δp
task_cols_S = sorted([c for c in S_don.columns if str(c).startswith(("Q12__","Q13__","Q14__","Q15__","Q16__","Q17__"))])
task_cols_L = sorted([c for c in L_logs.columns if c in task_cols_S])
if task_cols_L:
    rows = []
    for c in task_cols_L:
        s = pd.to_numeric(S_don[c], errors="coerce").fillna(0).astype(int)
        l = pd.to_numeric(L_logs[c], errors="coerce").fillna(0).astype(int)
        p1, p0 = s.mean(), l.mean()
        rows.append((abs(p1-p0), p1-p0, c))
    rows.sort(reverse=True)
    best = rows[:5]
    print("Tasks: top |Δp| (pp):")
    for _, dp, c in best:
        print(f"  {c}: Δp={dp*100:.1f} pp")

# figures exist check
figs_63 = [
    "fig_6_3_Q7_violin_ecdf.pdf","fig_6_3_Q8_violin_ecdf.pdf","fig_6_3_Q9_violin_ecdf.pdf",
    "fig_6_3_Q10Q11_dots.pdf","fig_6_3_tasks_dots.pdf","fig_6_3_forest_allitems.pdf"
]
exist_ok = all(file_exists_nonzero(FIG/f) for f in figs_63)
print("6.3 figures exist:", ok(exist_ok))

# =========================================
# 6.4 — MODELS Q19/Q20 (existence checks)
# =========================================
print("\n== 6.4 Models ==")
tex_q19 = (TAB/"H6_4_q19_ordlogit.tex").exists()
tex_q20 = (TAB/"H6_4_q20_ridgelogit.tex").exists()
fig_q19 = file_exists_nonzero(FIG/"fig_6_4_q19_probs.pdf")
fig_q20 = file_exists_nonzero(FIG/"fig_6_4_q20_probs.pdf")

print(f"Table q19 exists: {ok(tex_q19)}")
print(f"Table q20 exists: {ok(tex_q20)}")
print(f"Figure q19 exists: {ok(fig_q19)}")
print(f"Figure q20 exists: {ok(fig_q20)}")

# =========================================
# SUMMARY
# =========================================
print("\n== SUMMARY ==")
print("Funnel:", ok(funnel_ok))
print("6.1 balance:", ok(bal_csv.exists()))
print("6.2 allitems & numeric:", ok(allitems_csv.exists() and usenum_csv.exists()))
print("6.3 figs:", ok(exist_ok))
print("6.4 figs+tables:", ok(tex_q19 and tex_q20 and fig_q19 and fig_q20))

