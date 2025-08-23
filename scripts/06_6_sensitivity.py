import pathlib, json, re, math
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER, RES = ROOT/"derived", ROOT/"results"
TAB = RES/"tables"; TAB.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def load_all():
    S_all  = pd.read_parquet(DER/"S_all.parquet")
    S_don  = pd.read_parquet(DER/"S_donors.parquet")
    L_logs = pd.read_parquet(DER/"L_logs.parquet")
    labels = json.load(open(DER/"labels_SurveyFull.json", encoding="utf-8"))
    return S_all, S_don, L_logs, labels

def field(labels, q):
    for col, lab in labels.items():
        if str(lab).startswith(q):
            return col
    return None

def parse_midpoint(val, mult=1.5):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower().replace("—","-").replace("–","-")
    s = (s.replace(" to ","-").replace("tot","-").replace(" per dag","").replace("/dag","")
           .replace("minutes","").replace("minute","").replace("mins","").replace("minuten","").replace("min","").strip())
    m = re.match(r"^\s*(\d+)\s*$", s)
    if m: return float(m.group(1))
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if m: return (float(m.group(1))+float(m.group(2)))/2.0
    m = re.match(r"^\s*>\s*(\d+)\s*$", s) or re.match(r"^\s*(\d+)\s*\+\s*$", s)
    if m: return float(m.group(1))*mult
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan

def hl_diff(xs, xl, B=3000, seed=42):
    xs = pd.to_numeric(xs, errors="coerce").dropna().values
    xl = pd.to_numeric(xl, errors="coerce").dropna().values
    if len(xs)==0 or len(xl)==0:
        return np.nan, np.nan, np.nan, np.nan
    med = float(np.median((xs.reshape(-1,1)-xl.reshape(1,-1)).ravel()))
    rng = np.random.default_rng(seed)
    samp = []
    for _ in range(B):
        xb = rng.choice(xs, size=len(xs), replace=True)
        yb = rng.choice(xl, size=len(xl), replace=True)
        samp.append(np.median((xb.reshape(-1,1)-yb.reshape(1,-1)).ravel()))
    lo, hi = float(np.percentile(samp,2.5)), float(np.percentile(samp,97.5))
    try:
        p = float(mannwhitneyu(xs, xl, alternative="two-sided").pvalue)
    except Exception:
        p = np.nan
    return med, lo, hi, p

def wilson_ci(k,n,alpha=0.05):
    if n==0: return (np.nan,np.nan)
    from scipy.stats import norm
    z = norm.ppf(1-alpha/2); p=k/n
    den = 1+z*z/n; center = p + z*z/(2*n)
    pm  = z*math.sqrt((p*(1-p) + z*z/(4*n))/n)
    lo=(center-pm)/den; hi=(center+pm)/den
    return (max(0.0,lo), min(1.0,hi))

# ---------- load ----------
S_all, S_don, L_logs, labels = load_all()
Q7f, Q8f, Q9f = field(labels,"Q7"), field(labels,"Q8"), field(labels,"Q9")

# 6.6.A — numeric sensitivity (midpoint multiplier 10+ -> x1.5)
def add_alt_mid(df):
    out = df.copy()
    if Q7f in out.columns: out["Q7_mid_alt"] = out[Q7f].apply(parse_midpoint)
    if Q8f in out.columns:
        out["Q8_mid_alt"]  = out[Q8f].apply(parse_midpoint)
        out["Q8w_mid_alt"] = out["Q8_mid_alt"]*7.0
    if Q9f in out.columns: out["Q9_mid_alt"] = out[Q9f].apply(parse_midpoint)
    return out

S_don_alt  = add_alt_mid(S_don)
L_logs_alt = add_alt_mid(L_logs)

rows = []
for q, base_col, alt_col in [
    ("Q7","Q7_mid","Q7_mid_alt"),
    ("Q8","Q8w_mid","Q8w_mid_alt"),
    ("Q9","Q9_mid","Q9_mid_alt"),
]:
    if base_col in S_don.columns and base_col in L_logs.columns and alt_col in S_don_alt.columns and alt_col in L_logs_alt.columns:
        b_med,b_lo,b_hi,b_p = hl_diff(S_don[base_col], L_logs[base_col])
        a_med,a_lo,a_hi,a_p = hl_diff(S_don_alt[alt_col], L_logs_alt[alt_col])
        rows.append({
            "Item": q,
            "HL_diff_baseline": b_med, "CI_low_baseline": b_lo, "CI_high_baseline": b_hi, "U_p_baseline": b_p,
            "HL_diff_alt": a_med, "CI_low_alt": a_lo, "CI_high_alt": a_hi, "U_p_alt": a_p,
            "Abs_change": (a_med - b_med) if (pd.notna(a_med) and pd.notna(b_med)) else np.nan
        })
sens_num = pd.DataFrame(rows)
for c in sens_num.columns:
    if c != "Item": sens_num[c] = pd.to_numeric(sens_num[c], errors="coerce")
sens_num.to_csv(TAB/"H6_6_numeric_sensitivity.csv", index=False)

def to_latex(df, cap, lab, path, colfmt=None):
    body = df.round(3).to_latex(index=False, float_format="%.3f", column_format=colfmt or None)
    txt = "\\begin{table}[t]\\centering\\small\\caption{"+cap+"}\\label{"+lab+"}\n"+body+"\n\\end{table}\n"
    path.write_text(txt, encoding="utf-8")

to_latex(sens_num, "Numeric sensitivity (Q7--Q9): baseline vs alternative midpoint multiplier (10+ → ×1.5).",
         "tab:6_6_numeric_sens", TAB/"H6_6_numeric_sensitivity.tex", colfmt="l r r r r r r r r r")

# 6.6.B — usage_index baseline vs alt (Q7+Q8 only): Q19/Q20 snapshots
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def label_col(pref):
    for c,l in labels.items():
        if str(l).startswith(pref) and c in S_all.columns:
            return c
    return None

Q19c, Q20c = label_col("Q19"), label_col("Q20")
PLANc, STATc = label_col("Q3"), label_col("Q5")

def zscore_cols(df, cols):
    x = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in cols if c in df.columns], axis=1)
    z = (x - x.mean())/x.std(ddof=0)
    return z.mean(axis=1)

S_all = S_all.copy()
S_all["usage_index_alt"] = zscore_cols(S_all, ["Q7_mid","Q8w_mid"])  # exclude Q9

def prep_q19(df, usage_col):
    d = pd.DataFrame({
        "y": df[Q19c], "usage": df[usage_col], "tasks": df["num_tasks"],
        "plan": df[PLANc], "status": df[STATc]
    })
    d = d.dropna()
    # ordinal scoring
    def score_q19(x):
        s = str(x).lower()
        if re.search(r"essential|indispens|crucial|onmisbaar|extreme", s): return 5
        if re.search(r"fairly|tamelijk|very|erg|zeer", s): return 4
        if re.search(r"somewhat|redelijk|moderate|gemidd", s): return 3
        if re.search(r"slight|beetje|enigszins", s): return 2
        if re.search(r"not at all|helemaal niet|geen|^not\\b|niet belangrijk", s): return 1
        m = re.search(r"\\b([1-5])\\b", s);  return int(m.group(1)) if m else np.nan
    d["y"] = d["y"].apply(score_q19).astype("float")
    d = d.dropna()
    if d.empty: return None, None
    X = pd.get_dummies(d[["usage","tasks","plan","status"]], columns=["plan","status"], drop_first=True)
    # force numeric float, drop non-finite and constant cols
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.replace([np.inf,-np.inf], np.nan).dropna(axis=1, how="any")
    nun = X.nunique(dropna=True)
    X = X.loc[:, nun > 1]
    y = d["y"].astype(int).values
    if X.shape[1]==0 or len(y)==0: return None, None
    return X, y

def ord_ORs(X, y):
    if X is None or y is None: return None
    mod = OrderedModel(y, X.values.astype(float), distr="logit")
    res = mod.fit(method="bfgs", disp=False)
    p = X.shape[1]
    params = pd.Series(res.params[-p:], index=X.columns)
    bse    = pd.Series(res.bse[-p:],    index=X.columns)
    OR  = np.exp(params); lo = np.exp(params-1.96*bse); hi = np.exp(params+1.96*bse)
    return pd.DataFrame({"Term":X.columns, "OR":OR, "CI_low":lo, "CI_high":hi}).round(3)

Xb, yb = prep_q19(S_all, "usage_index")
Xa, ya = prep_q19(S_all, "usage_index_alt")
ORb = ord_ORs(Xb, yb)
ORa = ord_ORs(Xa, ya)

def ridge_ORs(df, usage_col):
    d = pd.DataFrame({
        "y": df[Q20c], "usage": df[usage_col], "tasks": df["num_tasks"],
        "plan": df[PLANc], "status": df[STATc]
    }).dropna()
    def ybin(x):
        s = str(x).lower()
        if any(t in s for t in ["yes","ja","keep using","would still use","definitely","probably yes"]): return 1
        if any(t in s for t in ["no","nee","probably not","would not","stop"]): return 0
        return np.nan
    d["y"] = d["y"].apply(ybin)
    d = d.dropna()
    if d.empty: return None
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    pre = ColumnTransformer([("num", StandardScaler(with_mean=False), ["usage","tasks"]),
                             ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["plan","status"])])
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(d[["usage","tasks","plan","status"]], d["y"])
    oh = pipe.named_steps["pre"].named_transformers_["cat"]
    names = ["usage_index (def)", "num_tasks (def)"] + oh.get_feature_names_out(["plan","status"]).tolist()
    OR = np.exp(pipe.named_steps["clf"].coef_.ravel())
    return pd.DataFrame({"Term":names, "OR":OR}).round(3)

Rb = ridge_ORs(S_all, "usage_index")
Ra = ridge_ORs(S_all, "usage_index_alt")

def join_tables(tb, ta, title, lab, path):
    if tb is None or ta is None:
        path.write_text("\\begin{table}[t]\\centering\\small\\caption{"+title+"}\\label{"+lab+"} No data.\\end{table}\n", encoding="utf-8"); return
    M = pd.merge(tb, ta, on="Term", suffixes=("_base","_alt"), how="outer")
    body = M.to_latex(index=False, float_format="%.3f")
    path.write_text("\\begin{table}[t]\\centering\\small\\caption{"+title+"}\\label{"+lab+"}\n"+body+"\n\\end{table}\n", encoding="utf-8")

join_tables(ORb, ORa, "Q19 (ordinal) sensitivity: ORs baseline vs usage_index_alt (Q7+Q8 only).",
            "tab:6_6_q19_sens", TAB/"H6_6_models_sensitivity_q19.tex")
join_tables(Rb, Ra, "Q20 (ridge) sensitivity: ORs baseline vs usage_index_alt (Q7+Q8 only).",
            "tab:6_6_q20_sens", TAB/"H6_6_models_sensitivity_q20.tex")
with open(TAB/"H6_6_models_sensitivity.tex","w",encoding="utf-8") as f:
    f.write("\\input{results/tables/H6_6_models_sensitivity_q19.tex}\n")
    f.write("\\input{results/tables/H6_6_models_sensitivity_q20.tex}\n")

# 6.6.C — Q12 options by status (Student vs Non-Student)
def label_col(pref):
    for c,l in labels.items():
        if str(l).startswith(pref) and c in S_all.columns:
            return c
    return None
STATc = label_col("Q5")
q12_cols = [c for c in S_all.columns if c.startswith("Q12_")]
rows = []
if STATc and q12_cols:
    sub = S_all[[STATc]+q12_cols].copy()
    sub["is_student"] = sub[STATc].astype(str).str.contains("student", case=False, na=False)
    for grp_name, grp_df in [("Student", sub[sub["is_student"]==True]), ("Non-Student", sub[sub["is_student"]==False])]:
        N = len(grp_df)
        for c in q12_cols:
            k = int(pd.to_numeric(grp_df[c], errors="coerce").fillna(0).sum())
            p = k/N if N else np.nan
            lo,hi = wilson_ci(k,N)
            rows.append({"Group": grp_name, "Option": c.replace("Q12_","").replace("_"," "),
                         "n": k, "N": N, "share": p, "CI_low": lo, "CI_high": hi})
SG = pd.DataFrame(rows)
if not SG.empty:
    SG["%"] = (SG["share"]*100).round(1); SG["CI low %"]=(SG["CI_low"]*100).round(1); SG["CI high %"]=(SG["CI_high"]*100).round(1)
    show = SG[["Group","Option","n","N","%","CI low %","CI high %"]]
    body = show.to_latex(index=False, float_format="%.1f")
    (TAB/"H6_6_subgroup_tasks.tex").write_text("\\begin{table}[t]\\centering\\small\\caption{Q12 options by status (Student vs Non-Student)}\\label{tab:6_6_subtasks}\n"+body+"\n\\end{table}\n", encoding="utf-8")
else:
    (TAB/"H6_6_subgroup_tasks.tex").write_text("\\begin{table}[t]\\centering\\small\\caption{Q12 options by status}No data.\\end{table}\n", encoding="utf-8")

print("Written:")
print("-", TAB/"H6_6_numeric_sensitivity.csv")
print("-", TAB/"H6_6_numeric_sensitivity.tex")
print("-", TAB/"H6_6_models_sensitivity.tex")
print("-", TAB/"H6_6_subgroup_tasks.tex")

