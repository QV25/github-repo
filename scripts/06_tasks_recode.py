import pathlib, json, re, unicodedata
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER  = ROOT / "derived"
RES  = ROOT / "results"
TAB  = RES / "tables"
for p in [DER, TAB]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------- helpers ----------------
def load_core():
    S_all  = pd.read_parquet(DER/"S_all.parquet")
    S_don  = pd.read_parquet(DER/"S_donors.parquet")
    L_logs = pd.read_parquet(DER/"L_logs.parquet")
    labels = json.load(open(DER/"labels_SurveyFull.json", encoding="utf-8"))
    return S_all, S_don, L_logs, labels

def pick_field_from_labels(labels, qcode):
    for col, lab in labels.items():
        if str(lab).startswith(qcode):
            return col
    return None

def find_col_in_df(df, target_field):
    if target_field in df.columns:
        return target_field
    if not target_field:
        return None
    tf_norm = re.sub(r'\s+', ' ', target_field.strip().lower())
    for c in df.columns:
        cn = re.sub(r'\s+', ' ', str(c).strip().lower())
        if cn == tf_norm or cn.startswith(tf_norm[:40]):
            return c
    return None

def norm(s: str):
    if pd.isna(s): return ""
    x = unicodedata.normalize("NFKC", str(s))
    x = re.sub(r'\s+', ' ', x).strip()
    return x

def present(cell: str, pattern: str):
    """Case-insensitive substring/regex match in normalized text."""
    if not cell: return False
    s = norm(cell).lower()
    return re.search(pattern, s, flags=re.IGNORECASE) is not None

def slug(s: str):
    t = re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')
    return t[:60]

def make_dummies(df, col, qcode, canon_map):
    """
    Per canonische optie een dummy: 1 als optie in de tekst voorkomt, anders 0.
    'I did not choose …' -> geen opties (alle dummies 0).
    """
    if not col or col not in df.columns:
        return []
    series = df[col].astype(str)
    # detecteer 'I did not choose …' zinnen
    not_chosen = series.str.contains(r"i did not choose", case=False, na=False)
    newcols = []
    base = qcode

    # maak dummies
    for label, rx in canon_map:
        cname = f"{base}_{slug(label)}"
        df[cname] = 0
        mask = series.apply(lambda v: present(v, rx))
        # forceer 0 als 'not chosen' gemeld is
        df.loc[mask & ~not_chosen, cname] = 1
        newcols.append(cname)

    return newcols

def share_table(df, cols, label):
    if not cols:
        return pd.DataFrame(columns=["option","n","N","share"])
    N = len(df)
    rows = []
    for c in cols:
        k = int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())
        rows.append({"option": c, "n": k, "N": N, "share": k/N if N else np.nan})
    out = pd.DataFrame(rows).sort_values("share", ascending=False)
    out.to_csv(TAB / f"{label}_shares.csv", index=False)
    return out

# ---------------- canonieke opties per vraag ----------------
# NB: we matchen op herkenbare stroken van de optie-tekst; case-insensitive regex.
CANON = {
    "Q12": [
        ("Writing & communication", r"writing\s*&\s*professional communication"),
        ("Brainstorming / fun",     r"brainstorming\s*&\s*personal ideas\s*/\s*fun"),
        ("Coding / programming",    r"coding\s*/\s*programming help"),
        ("Language / translation",  r"language practice\s*or\s*translation"),
        ("Study / exam prep",       r"study revision\s*/\s*exam prep"),
        ("Other",                   r"\bother\b"),
    ],
    "Q13": [
        ("Outline ideas / slides",  r"outlining ideas|slides"),
        ("Draft full text",         r"drafting full text"),
        ("Proofread / tone",        r"proof-?reading|tone adjustment"),
        ("Summarise sources/notes", r"summarising sources|meeting notes"),
        ("Adjust style (audiences)",r"adjusting style.*audiences"),
    ],
    "Q14": [
        ("Academic/research",       r"academic|research topics"),
        ("Business/marketing",      r"business|marketing concepts"),
        ("Creative role-play",      r"creative role-?play.*jokes.*stories|creative role-?play"),
        ("What-if scenarios",       r"hypothetical.*what-?if"),
        ("Recommendations media",   r"recommendations.*books.*movies.*music|recommendations"),
        ("Trivia / knowledge",      r"trivia|general knowledge"),
        ("Other",                   r"\bother\b"),
    ],
    "Q15": [
        ("Generate code",           r"generating new code snippets|generate.*code"),
        ("Debug code",              r"debugging existing code|debug"),
        ("Explain code/concepts",   r"explaining code|explain.*concepts"),
        ("Convert between langs",   r"converting code between languages|convert"),
        ("Write unit tests",        r"writing unit tests|unit tests"),
        ("Other",                   r"\bother\b"),
    ],
    "Q16": [
        ("Translate full texts",    r"translating full texts.*languages|translate"),
        ("Improve grammar/style",   r"improving grammar|style.*target language"),
        ("Vocabulary drills",       r"vocabulary drills|word lists"),
        ("Pronunciation guidance",  r"pronunciation|phonetic guidance"),
        ("Conversation / dialogue", r"conversational practice|dialogue role-?play"),
        ("Other",                   r"\bother\b"),
    ],
    "Q17": [
        ("Summarise lectures/readings", r"summarising lecture notes|readings"),
        ("Generate practice Qs/quizzes",r"generating practice questions|quizzes"),
        ("Explain difficult concepts",  r"explaining difficult concepts"),
        ("Review flashcards/terms",     r"reviewing flashcards|key terms"),
        ("Other",                       r"\bother\b"),
    ],
}

# ---------------- run ----------------
print("== step 2: recode tasks into per-option dummies (Q12–Q17) ==")
S_all, S_don, L_logs, labels = load_core()

# locate source columns
def locate_cols(df):
    out = {}
    for q in ["Q12","Q13","Q14","Q15","Q16","Q17"]:
        field = pick_field_from_labels(labels, q)
        out[q] = find_col_in_df(df, field)
    return out

cols_S = locate_cols(S_all)
cols_D = locate_cols(S_don)
cols_L = locate_cols(L_logs)

# build dummies per dataset
created = {"S_all": [], "S_don": [], "L_logs": []}
for name, df, cols in [("S_all", S_all, cols_S), ("S_don", S_don, cols_D), ("L_logs", L_logs, cols_L)]:
    for qcode, canon_map in CANON.items():
        newcols = make_dummies(df, cols[qcode], qcode, canon_map)
        created[name].extend(newcols)

# recompute task counts
Q12_cols_S = [c for c in created["S_all"] if c.startswith("Q12_")]
Q12_cols_D = [c for c in created["S_don"] if c.startswith("Q12_")]
Q12_cols_L = [c for c in created["L_logs"] if c.startswith("Q12_")]

S_all["num_tasks_main"] = pd.DataFrame({c: pd.to_numeric(S_all.get(c), errors="coerce").fillna(0) for c in Q12_cols_S}).sum(axis=1).astype("Int64") if Q12_cols_S else np.nan
S_don["num_tasks_main"] = pd.DataFrame({c: pd.to_numeric(S_don.get(c), errors="coerce").fillna(0) for c in Q12_cols_D}).sum(axis=1).astype("Int64") if Q12_cols_D else np.nan
L_logs["num_tasks_main"] = pd.DataFrame({c: pd.to_numeric(L_logs.get(c), errors="coerce").fillna(0) for c in Q12_cols_L}).sum(axis=1).astype("Int64") if Q12_cols_L else np.nan

# behou oude num_tasks, maar maak ook 'num_tasks_main' expliciet; voor modellen kunnen we later kiezen welke we gebruiken
# (optioneel) update 'num_tasks' om gelijk te zijn aan main tasks:
S_all["num_tasks"] = S_all["num_tasks_main"]
S_don["num_tasks"] = S_don["num_tasks_main"]
L_logs["num_tasks"] = L_logs["num_tasks_main"]

# write updated parquet (in-place, zodat downstream scripts ze meteen gebruiken)
S_all.to_parquet(DER/"S_all.parquet", index=False)
S_don.to_parquet(DER/"S_donors.parquet", index=False)
L_logs.to_parquet(DER/"L_logs.parquet", index=False)

# shares overzicht (voor snelle sanity)
share_table(S_all, Q12_cols_S, "Q12_main_tasks_survey")
share_table(S_don, Q12_cols_D, "Q12_main_tasks_donor")
share_table(L_logs, Q12_cols_L, "Q12_main_tasks_logs")

# codebook opslaan (final)
rows = []
for q, opts in CANON.items():
    for label, rx in opts:
        rows.append({"question": q, "canonical": label, "regex": rx})
pd.DataFrame(rows).to_csv(DER/"tasks_codebook_final.csv", index=False)

print("\ncreated dummies (counts):")
print("S_all:", len(created["S_all"]), "columns")
print("S_don:", len(created["S_don"]), "columns")
print("L_logs:", len(created["L_logs"]), "columns")
print("\nwritten:")
print("-", DER/"S_all.parquet")
print("-", DER/"S_donors.parquet")
print("-", DER/"L_logs.parquet")
print("-", TAB/"Q12_main_tasks_survey_shares.csv")
print("-", TAB/"Q12_main_tasks_donor_shares.csv")
print("-", TAB/"Q12_main_tasks_logs_shares.csv")
print("-", DER/"tasks_codebook_final.csv")

