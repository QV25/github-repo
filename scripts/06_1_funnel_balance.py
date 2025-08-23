import json, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER  = ROOT / "derived"
RES  = ROOT / "results"
TAB  = RES / "tables"
FIG  = RES / "figures"
for p in [TAB, FIG]:
    p.mkdir(parents=True, exist_ok=True)

# ---- load processed ----
S_all  = pd.read_parquet(DER / "S_all.parquet")
L_logs = pd.read_parquet(DER / "L_logs.parquet")
labels = json.load(open(DER / "labels_SurveyFull.json", encoding="utf-8"))

def col_by_label_prefix(prefix: str) -> str | None:
    """Vind de kolomnaam in S_all waarvan het label met 'Q#' begint (bv. 'Q1')."""
    for col, lab in labels.items():
        if col in S_all.columns and str(lab).startswith(prefix):
            return col
    return None

# ---- 6.1.A Funnel ----
funnel = pd.DataFrame([
    {"step": "Survey responses",       "n": int(len(S_all))},
    {"step": "Survey donors (subset)", "n": int(S_all.get("is_donor", pd.Series([], dtype=bool)).sum())},
    {"step": "Logs-derived donors",    "n": int(len(L_logs))},
])
funnel_path = TAB / "table_6_1_funnel.csv"
funnel.to_csv(funnel_path, index=False)

print("Funnel:")
print(funnel.to_string(index=False))

# ---- 6.1.B Balance donors vs non-donors (SMD) ----
# Categorie-variabelen uit labels (Q1 gender, Q2 age band, Q3 plan, Q5 status)
cols = {
    "gender":   col_by_label_prefix("Q1"),
    "age_band": col_by_label_prefix("Q2"),
    "plan":     col_by_label_prefix("Q3"),
    "status":   col_by_label_prefix("Q5"),
}

S_all["is_donor"] = S_all["is_donor"].astype(bool)
g1 = S_all[S_all["is_donor"]]      # donors
g0 = S_all[~S_all["is_donor"]]     # non-donors

def smd_binary(x1: pd.Series, x0: pd.Series) -> float:
    p1 = np.nanmean(x1.astype(float))
    p0 = np.nanmean(x0.astype(float))
    p  = (p1 + p0) / 2.0
    den = np.sqrt(p * (p - 1.0) * -1.0) if p not in (0.0, 1.0) else np.nan  # sqrt(p*(1-p))
    if isinstance(den, float) and (den == 0 or np.isnan(den)):
        return np.nan
    return (p1 - p0) / den

def smd_cont(x1: pd.Series, x0: pd.Series) -> float:
    x1 = pd.to_numeric(x1, errors="coerce")
    x0 = pd.to_numeric(x0, errors="coerce")
    n1, n0 = x1.notna().sum(), x0.notna().sum()
    mu1, mu0 = x1.mean(), x0.mean()
    s1, s0   = x1.std(ddof=1), x0.std(ddof=1)
    sp_num   = max(n1 - 1, 0) * (s1 ** 2) + max(n0 - 1, 0) * (s0 ** 2)
    sp_den   = max(n1 + n0 - 2, 1)
    sp       = np.sqrt(sp_num / sp_den) if sp_den > 0 else np.nan
    if not (isinstance(sp, float) and sp > 0):
        return np.nan
    return (mu1 - mu0) / sp

rows = []

# Categoricals: maak one-hot en bereken SMD per level
for var, col in cols.items():
    if not col:
        rows.append({"variable": var, "level": "<not found>", "SMD": np.nan})
        continue
    # Missing expliciet als categorie meenemen
    series = S_all[col].astype("object").fillna("Missing").astype("category")
    dummies = pd.get_dummies(series, prefix=var, drop_first=False)
    for dcol in dummies.columns:
        s1 = dummies.loc[g1.index, dcol].astype(float)
        s0 = dummies.loc[g0.index, dcol].astype(float)
        level = dcol.split("_", 1)[1] if "_" in dcol else dcol
        rows.append({"variable": var, "level": level, "SMD": smd_binary(s1, s0)})

# Continue: usage_index
if "usage_index" not in S_all.columns:
    S_all["usage_index"] = np.nan
rows.append({
    "variable": "usage_index",
    "level": "(continuous)",
    "SMD": smd_cont(g1["usage_index"], g0["usage_index"])
})

balance = pd.DataFrame(rows)
balance["absSMD"] = balance["SMD"].abs()
balance = balance.sort_values(["variable", "absSMD"], ascending=[True, False])

balance_path = TAB / "table_6_1_balance_smd.csv"
balance.to_csv(balance_path, index=False)

print("\nBalance SMD (top 12 by |SMD|):")
print(balance.head(12).to_string(index=False))

print("\nGeschreven:")
print("-", funnel_path)
print("-", balance_path)

