import pathlib, json, re, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER  = ROOT / "derived"
RES  = ROOT / "results"
TAB  = RES / "tables"
FIG  = RES / "figures"
for p in [TAB, FIG]: p.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def load_data():
    S = pd.read_parquet(DER/"S_all.parquet")
    labels = json.load(open(DER/"labels_SurveyFull.json", encoding="utf-8"))
    return S, labels

def col_by_label_prefix(df, labels, prefix):
    for col, lab in labels.items():
        if col in df.columns and str(lab).startswith(prefix):
            return col
    return None

def simplify_plan(x: str) -> str:
    if pd.isna(x): return np.nan
    s = str(x).lower()
    if "plus" in s:  return "Plus"
    if "free" in s:  return "Free"
    return "Other"

YES_TOKENS = {"yes","ja","yep","definitely","probably yes","would still use","keep using"}
NO_TOKENS  = {"no","nee","nope","probably not","would not","wouldn't","stop using"}
def q20_to_binary(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if any(tok in s for tok in YES_TOKENS): return 1
    if any(tok in s for tok in NO_TOKENS):  return 0
    if s in {"yes","no"}: return 1 if s=="yes" else 0
    return np.nan

ORDER_KEYS = [
    (r"not at all|geen|helemaal niet", 0),
    (r"^not\b|niet belangrijk", 1),
    (r"slight|beetje|enigszins", 2),
    (r"somewhat|redelijk|moderate|gemidd", 3),
    (r"fairly|tamelijk|very|erg|zeer", 4),
    (r"essential|indispens|crucial|onmisbaar|extreme", 5),
]
def score_q19_label(x: str):
    if pd.isna(x): return np.nan
    s = str(x).lower()
    for pat, val in ORDER_KEYS:
        if re.search(pat, s): return val
    m = re.search(r"\b([1-5])\b", s)
    if m: return int(m.group(1)) - 1
    return np.nan

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    m, sd = s.mean(skipna=True), s.std(skipna=True, ddof=0)
    if sd == 0 or np.isnan(sd): sd = 1.0
    return (s - m) / sd

def write_latex_table(df: pd.DataFrame, caption: str, label: str, path: pathlib.Path, colfmt=None, index=False):
    body = df.to_latex(index=index, escape=True, longtable=False, bold_rows=False,
                       na_rep="", float_format="%.3f", column_format=colfmt, buf=None)
    wrapped = ["\\begin{table}[t]","\\centering",f"\\caption{{{caption}}}",f"\\label{{{label}}}","\\small",body,"\\end{table}"]
    path.write_text("\n".join(wrapped), encoding="utf-8")

def drop_constant_cols(X: pd.DataFrame):
    const_mask = (X.nunique(dropna=False) <= 1) | (X.apply(lambda c: np.nanstd(c.values)==0))
    dropped = X.columns[const_mask].tolist()
    return X.loc[:, ~const_mask], dropped

# ---------- load & prepare ----------
S, labels = load_data()

Q19_col = col_by_label_prefix(S, labels, "Q19")
Q20_col = col_by_label_prefix(S, labels, "Q20")
PLAN_col = col_by_label_prefix(S, labels, "Q3")
STAT_col = col_by_label_prefix(S, labels, "Q5")
if not Q19_col or not Q20_col or not PLAN_col or not STAT_col:
    raise SystemExit("Missing required columns (Q19, Q20, Q3 plan, Q5 status).")

df = pd.DataFrame({
    "Q19_raw": S[Q19_col],
    "Q20_raw": S[Q20_col],
    "plan_raw": S[PLAN_col],
    "status_raw": S[STAT_col],
    "usage_index": S.get("usage_index", np.nan),
    # belangrijk: num_tasks is in stap 2 geüpdatet naar Q12-optie-som
    "num_tasks": S.get("num_tasks", np.nan),
})

df["plan"]     = df["plan_raw"].apply(simplify_plan)
df["status"]   = df["status_raw"].astype("object")
df["usage_z"]  = zscore(df["usage_index"])
df["tasks_z"]  = zscore(df["num_tasks"])
df["Q19_ord"]  = df["Q19_raw"].apply(score_q19_label)
df["Q20_bin"]  = df["Q20_raw"].apply(q20_to_binary)

# ===== Q19 — Ordinal Logit (HC3 if available; fallback conventional) =====
q19_df = df.dropna(subset=["Q19_ord","usage_z","tasks_z","plan","status"]).copy()
X_q19 = pd.get_dummies(q19_df[["usage_z","tasks_z","plan","status"]],
                       columns=["plan","status"], drop_first=True)
y_q19 = q19_df["Q19_ord"].astype(int)

X_q19 = X_q19.apply(pd.to_numeric, errors="coerce").astype(float)
rowmask = (~np.isnan(X_q19).any(axis=1)) & y_q19.notna()
X_q19 = X_q19.loc[rowmask]
y_q19 = y_q19.loc[rowmask].astype(int)
X_q19, dropped = drop_constant_cols(X_q19)

print(f"Q19 modeling N={len(y_q19)}, p={X_q19.shape[1]} (dropped constants: {dropped})")
if X_q19.shape[1] == 0:
    raise SystemExit("All predictors constant; cannot fit ordinal model.")

ord_mod = OrderedModel(endog=y_q19.values, exog=X_q19.values, distr="logit")
ord_res = ord_mod.fit(method="bfgs", disp=False)

# robust SE fallback
robust_used = True
try:
    ord_rob = ord_res.get_robustcov_results(cov_type="HC3")
    if ord_rob is None:
        robust_used = False
        ord_rob = ord_res
except Exception:
    robust_used = False
    ord_rob = ord_res

# slope terms = last p entries
p = X_q19.shape[1]
params = pd.Series(ord_rob.params[-p:], index=[f"x{i}" for i in range(p)])
bse    = pd.Series(ord_rob.bse[-p:],    index=[f"x{i}" for i in range(p)])
pvals  = pd.Series(ord_rob.pvalues[-p:], index=[f"x{i}" for i in range(p)])
name_map = {f"x{i}": col for i, col in enumerate(X_q19.columns)}

OR  = np.exp(params)
lo  = np.exp(params - 1.96*bse)
hi  = np.exp(params + 1.96*bse)

caption_q19 = ("Q19 (importance): ordinal logit with HC3 robust SE."
               if robust_used else
               "Q19 (importance): ordinal logit with conventional SE (HC3 not available here).")

q19_tab = pd.DataFrame({
    "Term": [str(name_map[k]).replace("^plan_", "plan: ").replace("^status_", "status: ") for k in OR.index],
    "OR": OR.values,
    "CI_low": lo.values,
    "CI_high": hi.values,
    "p_value": pvals.values
})
# prettify names
q19_tab["Term"] = q19_tab["Term"].str.replace("^plan_", "plan: ", regex=True)\
                                 .str.replace("^status_", "status: ", regex=True)\
                                 .str.replace("usage_z","usage_index (z)", regex=False)\
                                 .str.replace("tasks_z","num_tasks (z)", regex=False)

write_latex_table(q19_tab.assign(OR=lambda d: d["OR"].round(3),
                                 CI_low=lambda d: d["CI_low"].round(3),
                                 CI_high=lambda d: d["CI_high"].round(3),
                                 p_value=lambda d: d["p_value"].round(3)),
                  caption=caption_q19,
                  label="tab:6_4_q19_ordlogit",
                  path=TAB/"H6_4_q19_ordlogit.tex",
                  colfmt="l r r r r", index=False)

# predicted prob (top category) vs usage percentile, per existing plan
status_mode = q19_df["status"].mode().iloc[0] if not q19_df["status"].mode().empty else q19_df["status"].dropna().unique()[0]
plan_levels = sorted(q19_df["plan"].dropna().unique().tolist())

grid = np.linspace(np.nanpercentile(df["usage_z"],5), np.nanpercentile(df["usage_z"],95), 80)

def predict_q19_top_prob(usage_z, plan_label, status_label):
    row = pd.DataFrame({"usage_z":[usage_z], "tasks_z":[q19_df["tasks_z"].mean()] , "plan":[plan_label], "status":[status_label]})
    xr = pd.get_dummies(row, columns=["plan","status"], drop_first=True)
    for c in X_q19.columns:
        if c not in xr.columns: xr[c] = 0.0
    xr = xr[X_q19.columns].apply(pd.to_numeric, errors="coerce").astype(float)
    probs = ord_res.model.predict(ord_res.params, exog=xr.values)  # shape (1, K)
    return float(probs[0, -1])  # highest category

fig, ax = plt.subplots(figsize=(7,4))
curves = {}
for pl in plan_levels:
    ys = [predict_q19_top_prob(u, pl, status_mode) for u in grid]
    curves[pl] = np.array(ys)
# draw with distinct linestyles, mark overlap
styles = ["-", "--", ":", "-."]
for i, pl in enumerate(plan_levels):
    label = f"Plan: {pl}"
    ax.plot(np.linspace(5,95,len(grid)), curves[pl], linestyle=styles[i % len(styles)], label=label)
# if exactly two and nearly identical, add "(overlap)"
if len(plan_levels) == 2:
    diff = np.max(np.abs(curves[plan_levels[0]] - curves[plan_levels[1]]))
    if diff < 1e-3:
        ax.legend_.remove() if ax.legend_ else None
        ax.plot([], [], linestyle=styles[0], label=f"Plan: {plan_levels[0]} / {plan_levels[1]} (overlap)")
ax.set_xlabel("Usage index percentile")
ax.set_ylabel("Pr(Q19 = highest category)")
ax.set_title("Q19 predicted probability by usage and plan")
ax.grid(linestyle=":", linewidth=0.7)
ax.legend()
fig.tight_layout()
fig.savefig(FIG/"fig_6_4_q19_probs.pdf", bbox_inches="tight")
plt.close(fig)

(TAB / "H6_4_fig_q19.tex").write_text(
    "\\begin{figure}[t]\\centering\\includegraphics[width=0.8\\linewidth]{figures/fig_6_4_q19_probs.pdf}"
    "\\caption{Q19 predicted probability for the highest importance category vs usage percentile, by plan (status held at mode).}"
    "\\label{fig:6_4_q19_probs}\\end{figure}\n",
    encoding="utf-8"
)

# ===== Q20 — Ridge Logistic (L2) =====
q20_df = df.dropna(subset=["Q20_bin","usage_z","tasks_z","plan","status"]).copy()
X_q20 = q20_df[["usage_z","tasks_z","plan","status"]].copy()
y_q20 = q20_df["Q20_bin"].astype(int)

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), ["usage_z","tasks_z"]),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["plan","status"])
    ],
    remainder="drop"
)
logit_ridge = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)
pipe = Pipeline([("pre", pre), ("clf", logit_ridge)])
pipe.fit(X_q20, y_q20)

oh = pipe.named_steps["pre"].named_transformers_["cat"]
num_names = ["usage_index (z)", "num_tasks (z)"]
cat_names = oh.get_feature_names_out(["plan","status"]).tolist()
feat_names = num_names + cat_names
coef = pipe.named_steps["clf"].coef_.ravel()
OR = np.exp(coef)
q20_tab = pd.DataFrame({"Term": feat_names, "OR": OR})
write_latex_table(q20_tab.assign(OR=lambda d: d["OR"].round(3)),
                  caption="Q20 (paid-only use): ridge logistic regression (L2). Odds ratios (no SE).",
                  label="tab:6_4_q20_ridgelogit",
                  path=TAB/"H6_4_q20_ridgelogit.tex",
                  colfmt="l r", index=False)

# predicted prob vs usage percentile per existing plan
status_mode2 = q20_df["status"].mode().iloc[0] if not q20_df["status"].mode().empty else q20_df["status"].dropna().unique()[0]
plan_levels2 = sorted(q20_df["plan"].dropna().unique().tolist())
grid2 = np.linspace(np.nanpercentile(df["usage_z"],5), np.nanpercentile(df["usage_z"],95), 80)

fig, ax = plt.subplots(figsize=(7,4))
curves2 = {}
for pl in plan_levels2:
    rows = pd.DataFrame({
        "usage_z": grid2,
        "tasks_z": q20_df["tasks_z"].mean(),
        "plan": [pl]*len(grid2),
        "status": [status_mode2]*len(grid2)
    })
    pr = pipe.predict_proba(rows)[:,1]
    curves2[pl] = pr
    ax.plot(np.linspace(5,95,len(grid2)), pr, linestyle=styles[plan_levels2.index(pl) % len(styles)], label=f"Plan: {pl}")
if len(plan_levels2) == 2:
    diff2 = np.max(np.abs(curves2[plan_levels2[0]] - curves2[plan_levels2[1]]))
    if diff2 < 1e-3:
        ax.legend_.remove() if ax.legend_ else None
        ax.plot([], [], linestyle=styles[0], label=f"Plan: {plan_levels2[0]} / {plan_levels2[1]} (overlap)")
ax.set_xlabel("Usage index percentile")
ax.set_ylabel("Pr(Q20 = yes)")
ax.set_title("Q20 predicted probability vs usage, by plan")
ax.grid(linestyle=":", linewidth=0.7)
ax.legend()
fig.tight_layout()
fig.savefig(FIG/"fig_6_4_q20_probs.pdf", bbox_inches="tight")
plt.close(fig)

(TAB / "H6_4_fig_q20.tex").write_text(
    "\\begin{figure}[t]\\centering\\includegraphics[width=0.8\\linewidth]{figures/fig_6_4_q20_probs.pdf}"
    "\\caption{Q20 predicted probability of continuing use if paid-only, vs usage percentile, by plan (status held at mode).}"
    "\\label{fig:6_4_q20_probs}\\end{figure}\n",
    encoding="utf-8"
)

print("Written:")
print("-", TAB/"H6_4_q19_ordlogit.tex")
print("-", TAB/"H6_4_q20_ridgelogit.tex")
print("-", TAB/"H6_4_fig_q19.tex")
print("-", TAB/"H6_4_fig_q20.tex")
print("-", FIG/"fig_6_4_q19_probs.pdf")
print("-", FIG/"fig_6_4_q20_probs.pdf")

