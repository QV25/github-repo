import pathlib, json, re, unicodedata
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER  = ROOT / "derived"
RES  = ROOT / "results"
TAB  = RES / "tables"
for p in [TAB]:
    p.mkdir(parents=True, exist_ok=True)

# -------- helpers --------
def load_core():
    S_all  = pd.read_parquet(DER/"S_all.parquet")
    S_don  = pd.read_parquet(DER/"S_donors.parquet")
    L_logs = pd.read_parquet(DER/"L_logs.parquet")
    labels = json.load(open(DER/"labels_SurveyFull.json", encoding="utf-8"))
    return S_all, S_don, L_logs, labels

def pick_field_from_labels(labels, qcode):
    """Pak de veldnaam (lange vraagtekst) bij bv. 'Q12' uit labels_SurveyFull.json."""
    for col, lab in labels.items():
        if str(lab).startswith(qcode):
            return col
    return None

def find_col_in_df(df, target_field):
    """Zoek de kolom in df die overeenkomt met de veldnaam uit SurveyFull-labels.
       Fallback: fuzzy match op beginsel van de veldnaam."""
    if target_field in df.columns:
        return target_field
    if not target_field:
        return None
    # normaliseer strings en probeer match op prefix
    tf_norm = re.sub(r'\s+', ' ', target_field.strip().lower())
    for c in df.columns:
        cn = re.sub(r'\s+', ' ', str(c).strip().lower())
        if cn == tf_norm or cn.startswith(tf_norm[:40]):
            return c
    return None

def normalize_token(tok: str):
    """Maak tokens consistent zonder betekenis te verliezen."""
    if tok is None:
        return None
    # unicode normalisatie
    s = unicodedata.normalize("NFKC", str(tok))
    s = s.strip()
    # verwijder afsluitende punten/kommas etc. alleen aan de randen
    s = re.sub(r'^[\s,;:/·•-]+|[\s,;:/·•-]+$', '', s)
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s)
    # niets naar lower/upper forceren; behoud originele casing
    return s

def split_selections(cell):
    """Qualtrics multiselect is (vrijwel altijd) ';'-gescheiden. Splits alleen daarop."""
    if pd.isna(cell):
        return []
    s = str(cell)
    # uniformeer ‘rare’ puntkomma's
    s = s.replace('；',';')
    parts = [normalize_token(p) for p in s.split(';')]
    return [p for p in parts if p]

def count_tokens_for(df, col):
    """Tel per uniek token het #respondenten dat het token selecteert."""
    if not col or col not in df.columns:
        return {}, 0
    n_valid = int(df[col].notna().sum())
    counts = {}
    for val in df[col].dropna():
        toks = split_selections(val)
        # per respondent per token max 1 tellen
        for t in set(toks):
            counts[t] = counts.get(t, 0) + 1
    return counts, n_valid

def write_token_table(qcode, all_counts):
    """
    all_counts: dict met keys 'survey','donor','logs' -> (counts_dict, n_valid)
    Schrijf CSV met kolommen: token, n_survey, n_donor, n_logs, n_all, frac_*
    """
    rows = {}
    n_s, n_d, n_l = all_counts["survey"][1], all_counts["donor"][1], all_counts["logs"][1]
    for source in ["survey","donor","logs"]:
        cnt, _ = all_counts[source]
        for tok, n in cnt.items():
            rows.setdefault(tok, {"token": tok, "n_survey":0,"n_donor":0,"n_logs":0})
            key = {"survey":"n_survey","donor":"n_donor","logs":"n_logs"}[source]
            rows[tok][key] = n
    out = pd.DataFrame(rows.values())
    if out.empty:
        out = pd.DataFrame(columns=["token","n_survey","n_donor","n_logs"])
    out["n_all"] = out[["n_survey","n_donor","n_logs"]].sum(axis=1)
    # fracties per bron (t.o.v. niet-missende rijen)
    out["frac_survey"] = out["n_survey"] / (n_s if n_s else np.nan)
    out["frac_donor"]  = out["n_donor"]  / (n_d if n_d else np.nan)
    out["frac_logs"]   = out["n_logs"]   / (n_l if n_l else np.nan)
    out = out.sort_values(["n_all","n_survey","n_logs","token"], ascending=[False,False,False,True])
    out.to_csv(TAB / f"{qcode}_tokens.csv", index=False)
    return out, n_s, n_d, n_l

# -------- run --------
print("== step 1: tokenize Q12–Q17 across survey/donor/logs ==")
S_all, S_don, L_logs, labels = load_core()

qcodes = ["Q12","Q13","Q14","Q15","Q16","Q17"]
fieldmap = {q: pick_field_from_labels(labels, q) for q in qcodes}

# vind kolomnamen in alle drie datasets
cols = {}
for q in qcodes:
    cols[q] = {
        "survey": find_col_in_df(S_all, fieldmap[q]),
        "donor":  find_col_in_df(S_don, fieldmap[q]),
        "logs":   find_col_in_df(L_logs, fieldmap[q]),
    }

# tel tokens + schrijf CSV's
all_codebook_rows = []
for q in qcodes:
    cnt_s, n_s = count_tokens_for(S_all, cols[q]["survey"])
    cnt_d, n_d = count_tokens_for(S_don, cols[q]["donor"])
    cnt_l, n_l = count_tokens_for(L_logs, cols[q]["logs"])
    table, ns, nd, nl = write_token_table(q, {"survey":(cnt_s,n_s),"donor":(cnt_d,n_d),"logs":(cnt_l,n_l)})
    # bouw seed codebook-rijen
    for tok in table["token"]:
        all_codebook_rows.append({"question": q, "raw_token": tok, "canonical": tok})

    # console top 15
    print(f"\n{q} column names → survey: {cols[q]['survey']!r}, donor: {cols[q]['donor']!r}, logs: {cols[q]['logs']!r}")
    print(f"{q} non-missing → survey N={ns}, donor N={nd}, logs N={nl}")
    top = table.head(15)[["token","n_survey","n_donor","n_logs","n_all"]]
    if not top.empty:
        print(f"{q} top tokens:")
        print(top.to_string(index=False))
    else:
        print(f"{q}: no tokens found.")

# schrijf seed codebook
codebook = pd.DataFrame(all_codebook_rows).drop_duplicates().sort_values(["question","raw_token"])
codebook.to_csv(DER / "tasks_codebook_seed.csv", index=False)
print(f"\nwritten:")
for q in qcodes:
    print("-", TAB / f"{q}_tokens.csv")
print("-", DER / "tasks_codebook_seed.csv")

