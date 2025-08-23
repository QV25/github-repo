import json, re, pathlib
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DERIVED = ROOT / "derived"
RESULTS = ROOT / "results"
for p in [DERIVED, RESULTS, RESULTS/"tables", RESULTS/"figures", RESULTS/"logs", DERIVED/"codebooks"]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def read_qualtrics_any(path: pathlib.Path):
    """
    Robuust inlezen van Qualtrics-export.
    Retourneert: df (kolommen = veldnamen/vragen-teksten), labels_dict {kolom -> 'Q#' label-tekst}.
    """
    # Probeer multi-index headers (labels + fields)
    try:
        df_multi = pd.read_csv(path, header=[0,1], low_memory=False, encoding="utf-8-sig")
        if isinstance(df_multi.columns, pd.MultiIndex):
            labels_top  = df_multi.columns.get_level_values(0).tolist()
            fieldnames  = df_multi.columns.get_level_values(1).tolist()
            labels_dict = {str(f): str(l) for f, l in zip(fieldnames, labels_top)}
            df = df_multi.copy()
            df.columns = fieldnames
            return df, labels_dict
    except Exception:
        pass
    # Fallback: header=1 en rij 0 als labels
    df = pd.read_csv(path, header=1, low_memory=False, encoding="utf-8-sig")
    head0 = pd.read_csv(path, header=None, nrows=1, low_memory=False, encoding="utf-8-sig")
    labels_top = head0.iloc[0].astype(str).tolist()
    if len(labels_top) != len(df.columns):
        labels_dict = {c: c for c in df.columns}
    else:
        labels_dict = {str(c): str(l) for c, l in zip(df.columns, labels_top)}
    return df, labels_dict

def find_col(df, candidates):
    cols = {re.sub(r'[^a-z0-9]+','', c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r'[^a-z0-9]+','', cand.lower())
        if key in cols:
            return cols[key]
    return None

def pick_primary_by_label(df, labels_dict, targets):
    """
    targets: list zoals ["Q7","Q8",...]
    return: {"Q7": kolomnaam_in_df, ...}
    """
    label_to_cols = {}
    for col in df.columns:
        lab = labels_dict.get(col, "")
        label_to_cols.setdefault(str(lab).strip(), []).append(col)
    out = {}
    for tq in targets:
        hit = None
        for lab, cols in label_to_cols.items():
            if lab.startswith(tq):
                hit = cols[0]; break
        out[tq] = hit
    return out

def parse_midpoint(val, open_end_multiplier=1.2):
    """
    Converteert categorie-tekst naar numeriek midpoint.
    Herkent o.a.: '0', '1-2'/'1–2', '3 to 5', '>10', '10+', '5-10 minutes', enz.
    """
    if pd.isna(val): return np.nan
    try:
        return float(val)
    except Exception:
        s = str(val).strip().lower()
        s = s.replace('—','-').replace('–','-')
        s = s.replace(' to ','-').replace('tot','-')
        # units/woorden verwijderen
        for tok in ['minutes','minute','mins','minuten','min','uur','hours','hour','h','m',
                    ' keer',' keer/dag',' per dag','/dag',' per week','/week']:
            s = s.replace(tok, '')
        s = s.replace('~','').replace('approx','').strip()

        if re.fullmatch(r'0+', s): return 0.0
        m = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s*$', s)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return (a + b) / 2.0
        m = re.match(r'^\s*>\s*(\d+)\s*$', s) or re.match(r'^\s*(\d+)\s*\+\s*$', s) or re.match(r'^\s*meer dan\s*(\d+)\s*$', s)
        if m:
            n = float(m.group(1)); return n * open_end_multiplier
        m = re.match(r'^\s*(\d+)', s)
        if m:
            return float(m.group(1))
        return np.nan

def midpoints_for(df, cols, multiplier=1.2):
    out = {}
    for c in cols:
        if c and c in df.columns:
            out[c+"_mid"] = df[c].apply(lambda v: parse_midpoint(v, multiplier))
    return out

YES_TOKENS = {"selected","ja","yes","true","waar","1","checked"}
NO_TOKENS  = {"no","nee","false","onwaar","0","not selected","notselected"}
def to_dummy(s):
    if pd.isna(s): return 0
    if isinstance(s,(int,float)) and not pd.isna(s):
        return 1 if float(s)!=0 else 0
    st = str(s).strip().lower()
    if st in YES_TOKENS or "selected" in st: return 1
    if st in NO_TOKENS: return 0
    if ";" in st:
        return 1 if any(x.strip() for x in st.split(";")) else 0
    return 1 if st else 0

def make_multiselect_dummies_with_labels(df, labels_dict, q_bases=("Q12","Q13","Q14","Q15","Q16","Q17")):
    df = df.copy()
    new_cols = []
    label_to_cols = {}
    for col in df.columns:
        lab = labels_dict.get(col, "")
        label_to_cols.setdefault(str(lab), []).append(col)

    for qb in q_bases:
        cols_for_q = []
        for lab, cols in label_to_cols.items():
            if str(lab).startswith(qb):
                cols_for_q.extend(cols)
        if not cols_for_q:
            continue
        if len(cols_for_q) == 1:
            col = cols_for_q[0]
            if df[col].astype(str).str.contains(';').any():
                opts = set()
                df[col].dropna().astype(str).str.split(';').apply(lambda xs: [opts.add(x.strip()) for x in xs])
                for opt in sorted(o for o in opts if o):
                    name = f"{qb}__{re.sub(r'[^A-Za-z0-9]+','_', opt).strip('_')}"
                    df[name] = df[col].astype(str).str.split(';').apply(lambda xs: int(opt in [x.strip() for x in xs]))
                    new_cols.append(name)
            else:
                name = f"{qb}__selected"
                df[name] = df[col].apply(to_dummy)
                new_cols.append(name)
        else:
            for col in cols_for_q:
                suffix = re.sub(r'[^A-Za-z0-9]+','_', col).strip('_')
                name = f"{qb}__{suffix}"
                df[name] = df[col].apply(to_dummy)
                new_cols.append(name)

    task_cols = [c for c in new_cols if re.match(r'^Q1[2-7]__', c)]
    if task_cols:
        df["num_tasks"] = df[task_cols].sum(axis=1).astype("Int64")
    return df, new_cols

def zscore_mean(cols_df):
    """
    Gemiddelde van z-scores per rij, zonder warnings:
    - per kolom alleen rekenen als er ≥2 geldige waarden zijn
    - rijen-gemiddelde met nansum/nancount (geen np.nanmean-warnings)
    """
    arr = cols_df.to_numpy(dtype=float)
    nrows, ncols = arr.shape
    z = np.full((nrows, ncols), np.nan, dtype=float)
    for j in range(ncols):
        col = arr[:, j]
        m = ~np.isnan(col)
        if m.sum() >= 2:
            mu = col[m].mean()
            sd = col[m].std(ddof=0)
            if sd == 0: sd = 1.0
            z[m, j] = (col[m] - mu) / sd
    # rijgemiddelde zonder warnings
    valid_counts = np.sum(~np.isnan(z), axis=1)
    row_sums = np.nansum(z, axis=1)
    out = np.full(nrows, np.nan, dtype=float)
    ok = valid_counts > 0
    out[ok] = row_sums[ok] / valid_counts[ok]
    return out

print("== stap 0: inlezen & standaardiseren ==")
S_all, labels_S = read_qualtrics_any(DATA/"Survey full.csv")
S_don, labels_D = read_qualtrics_any(DATA/"Survey Data log full.csv")
L_logs, labels_L = read_qualtrics_any(DATA/"Data logs.csv")

# ResponseId & donor-vlag
resp_col_S = find_col(S_all, ["ResponseId","Response ID","ResponseID","id","ID"])
resp_col_D = find_col(S_don, ["ResponseId","Response ID","ResponseID","id","ID"])
donor_ids = set(S_don[resp_col_D].dropna().astype(str).unique()) if resp_col_D else set()
S_all["is_donor"] = S_all[resp_col_S].astype(str).isin(donor_ids) if resp_col_S else False

print(f"S_all shape: {S_all.shape} | donor-vlaggen in Survey full: {int(S_all['is_donor'].sum()) if 'is_donor' in S_all else 0}")
print(f"S_don shape: {S_don.shape} | L_logs shape: {L_logs.shape}")

# Q7–Q11 & Q19–Q20 via labels
prim_S = pick_primary_by_label(S_all, labels_S, ["Q7","Q8","Q9","Q10","Q11","Q19","Q20"])
prim_D = pick_primary_by_label(S_don, labels_D, ["Q7","Q8","Q9","Q10","Q11","Q19","Q20"])
prim_L = pick_primary_by_label(L_logs, labels_L, ["Q7","Q8","Q9","Q10","Q11"])

# midpoints + alias naar Q#_mid
for df, prim in [(S_all, prim_S), (S_don, prim_D), (L_logs, prim_L)]:
    mids = midpoints_for(df, [prim.get(q) for q in ["Q7","Q8","Q9","Q10","Q11"]], multiplier=1.2)
    for name, series in mids.items():
        df[name] = series
    for q in ["Q7","Q8","Q9","Q10","Q11"]:
        src = prim.get(q)
        if src and (src + "_mid") in df.columns:
            df[f"{q}_mid"] = df[src + "_mid"]
    if "Q8_mid" in df.columns:
        df["Q8w_mid"] = df["Q8_mid"] * 7.0

# alias-kolommen voor Q19/Q20
for df, prim in [(S_all, prim_S), (S_don, prim_D)]:
    for q in ["Q19","Q20"]:
        col = prim.get(q)
        if col and col in df.columns:
            df[q] = df[col]

# multiselect dummies Q12–Q17
S_all, dums_S = make_multiselect_dummies_with_labels(S_all, labels_S)
S_don, dums_D = make_multiselect_dummies_with_labels(S_don, labels_D)
L_logs, dums_L = make_multiselect_dummies_with_labels(L_logs, labels_L)

# usage_index (gemiddelde z-scores van Q7_mid, Q8w_mid, Q9_mid) — alleen als >=2 aanwezig
def add_usage_index(df):
    cols = [c for c in ["Q7_mid","Q8w_mid","Q9_mid"] if c in df.columns]
    if len(cols) >= 2:
        df["usage_index"] = zscore_mean(df[cols])
    else:
        df["usage_index"] = np.nan
for df in (S_all, S_don, L_logs):
    add_usage_index(df)

# QC: missings
def missing_report(df, cols, name):
    cols_exist = [c for c in cols if c in df.columns]
    if not cols_exist:
        return None
    rep = pd.DataFrame({
        "variable": cols_exist,
        "n_missing": [int(df[c].isna().sum()) for c in cols_exist],
        "n_total": [int(df.shape[0]) for _ in cols_exist],
    })
    rep["pct_missing"] = (rep["n_missing"] / rep["n_total"]).round(4)
    out = RESULTS/"tables"/f"qc_missingness_{name}.csv"
    rep.to_csv(out, index=False)
    return rep

core_S = ["Q7_mid","Q8_mid","Q8w_mid","Q9_mid","usage_index","num_tasks","is_donor","Q19","Q20"]
core_D = ["Q7_mid","Q8_mid","Q8w_mid","Q9_mid","usage_index","num_tasks","Q19","Q20"]
core_L = ["Q7_mid","Q8_mid","Q8w_mid","Q9_mid","usage_index","num_tasks"]

rep_S = missing_report(S_all, core_S, "survey_full")
rep_D = missing_report(S_don, core_D, "donor_survey")
rep_L = missing_report(L_logs, core_L, "logs")

# save
S_all.to_parquet(DERIVED/"S_all.parquet", index=False)
S_don.to_parquet(DERIVED/"S_donors.parquet", index=False)
L_logs.to_parquet(DERIVED/"L_logs.parquet", index=False)
with open(DERIVED/"labels_SurveyFull.json","w",encoding="utf-8") as f:
    json.dump(labels_S, f, ensure_ascii=False, indent=2)

# console overzicht
def short(d, keys):
    return {k: d.get(k) for k in keys}
print("\n== detectie-overzicht (via labels) ==")
print(f"S_all primair: {short(prim_S, ['Q7','Q8','Q9','Q10','Q11','Q19','Q20'])}")
print(f"S_don primair: {short(prim_D, ['Q7','Q8','Q9','Q10','Q11','Q19','Q20'])}")
print(f"L_logs primair: {short(prim_L, ['Q7','Q8','Q9','Q10','Q11'])}")

print(f"\nS_all dummies Q12–Q17: {len(dums_S)}")
print(f"S_don dummies Q12–Q17: {len(dums_D)}")
print(f"L_logs dummies Q12–Q17: {len(dums_L)}")

for name, rep in [("survey_full", rep_S), ("donor_survey", rep_D), ("logs", rep_L)]:
    if rep is not None:
        print(f"\nmissings {name} (top 10):")
        print(rep.sort_values('pct_missing', ascending=False).head(10).to_string(index=False))

print("\nBESTANDEN GESCHREVEN:")
print(f"- {DERIVED/'S_all.parquet'}")
print(f"- {DERIVED/'S_donors.parquet'}")
print(f"- {DERIVED/'L_logs.parquet'}")
print(f"- {RESULTS/'tables'/'qc_missingness_survey_full.csv'} (en donor_survey/logs)")
print(f"- {DERIVED/'labels_SurveyFull.json'}")

