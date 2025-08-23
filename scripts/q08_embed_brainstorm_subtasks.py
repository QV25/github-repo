#!/usr/bin/env python3
"""
q08_embed_brainstorm_subtasks.py — hybride (embeddings + mini LLM refinement)

Analyseert *alleen donors die in q06 'Brainstorming & personal ideas - fun' kozen*.

Output → results/answers_q08.xlsx
"""
import os, sys, pathlib, json, datetime, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist

# ------------ project-modules ------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from embed_cache import EmbedCache
from llm_utils import gpt  # mini LLM refinement
# -----------------------------------------------------------------------

PARSED   = pathlib.Path("parsed/all.jsonl")
PROTOS   = pathlib.Path("prototypes_q08.json")
A6_XLS   = pathlib.Path("results/answers_q06.xlsx")
OUT_XLS  = pathlib.Path("results/answers_q08.xlsx")
CACHE_NS = "q08_brainstorm_sub"

# lees categorieën uit prototypes (volgorde = 1..K)
CATEGORIES = json.loads(PROTOS.read_text())
CATEGORIES = [str(x) for x in CATEGORIES]
mapping = {i+1: c for i, c in enumerate(CATEGORIES)}

# probeer index voor 'niet gekozen' te vinden (EN/NL)
def _find_none_idx(categories):
    for i, c in enumerate(categories, start=1):
        s = str(c).lower()
        if "did not choose" in s or "niet gekozen" in s:
            return i
    return None
NONE_IDX = _find_none_idx(CATEGORIES)

# Thresholds + model
THRESH    = 0.35
BAND_LOW  = 0.345
BAND_HIGH = 0.355
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-mini")

# Significantie-drempels per donor
MIN_N     = 3
MIN_SHARE = 0.05

# ------------ helpers ------------
def _extract_json_array_indices(s: str, n_max: int) -> list[int]:
    """Pak eerste JSON array uit tekst; filter op geldige 1..n indices."""
    try:
        m = re.search(r"\[[\s\S]*?\]", s)
        if not m:
            return []
        arr = json.loads(m.group(0))
        out = []
        for i in arr:
            ii = int(i)
            if 1 <= ii <= n_max and ii not in out:
                out.append(ii)
        return out
    except Exception:
        return []

def refine_with_llm(prompt_text: str, categories: list[str], none_idx: int|None) -> list[int]:
    """
    Vraag compacte disambiguation: alle toepasselijke sub-categorieën (1-based).
    Als 'none_idx' samen met andere labels voorkomt → verwijder 'none_idx'.
    """
    none_hint = f"If none apply, return [{none_idx}]." if none_idx else ""
    ask = (
        "Assign ALL applicable sub-categories for the user's prompt from this fixed list. "
        "Return ONLY a JSON array of 1-based indices (e.g., [1,3]). "
        f"{none_hint}\n\n"
        f"Categories (1..{len(categories)}): {json.dumps(categories, ensure_ascii=False)}\n\n"
        f"Prompt:\n{prompt_text}"
    )
    try:
        ans = gpt(ask, model=CHAT_MODEL, temperature=0, max_tokens=80)
    except Exception:
        return []
    out = _extract_json_array_indices(str(ans), len(categories))
    if none_idx and (none_idx in out) and (len(out) > 1):
        out = [i for i in out if i != none_idx]
    return out

def _canon_label(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("—", "-").replace("–", "-").replace("“", '"').replace("”", '"')
    # unify separators: "/" or "-" -> " - "
    s = re.sub(r"\s*[/\-]\s*", " - ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _brainstorm_donors_from_q06() -> set[str]:
    """Donors die exact de hoofdcategorie Brainstorming in q06 hebben (split op '|')."""
    if not A6_XLS.exists():
        raise FileNotFoundError("answers_q06.xlsx niet gevonden. Run eerst q06_embed.py.")
    a6 = pd.read_excel(A6_XLS)
    target = _canon_label("Brainstorming & personal ideas - fun")
    def has_brain(cat: str) -> bool:
        parts = [_canon_label(x) for x in str(cat).split("|")]
        return target in parts
    mask = a6["category"].astype(str).map(has_brain)
    donors = set(a6.loc[mask, "donor_id"].astype(str))
    return donors

# 0) donorset beperken via q06 -------------------------------------------
brain_donors = _brainstorm_donors_from_q06()
if not brain_donors:
    print("⚠︎ Geen donors met Brainstorming in q06. Exporteer lege sheet met alleen headers.")
    OUT_XLS.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(columns=["donor_id","category","survey_question","timestamp"]).to_excel(OUT_XLS, index=False)
    raise SystemExit(0)

# 1) prototypes → embeddings
ec          = EmbedCache(CACHE_NS)
proto_vecs  = ec.get_embeddings(CATEGORIES)          # (K, D)

# 2) prompts inlezen (ALLEEN geselecteerde donors)
df  = pd.read_json(PARSED, lines=True)
df  = df[df["donor_id"].astype(str).isin(brain_donors)].copy()

if df.empty:
    print("⚠︎ geen rijen over na donor-filter; output wordt leeg Excel met alleen headers.")
    valid = pd.DataFrame(columns=["donor_id","category","survey_question","timestamp"])
    OUT_XLS.parent.mkdir(exist_ok=True, parents=True)
    valid.to_excel(OUT_XLS, index=False)
    print(f"✅ klaar → {OUT_XLS}")
    raise SystemExit(0)

df["prompt_short"] = df["question"].str.slice(0, 400)
dfu = (
    df.drop_duplicates("prompt_short")
      .dropna(subset=["prompt_short"])
      .copy()
)
dfu["prompt_short"] = (
    dfu["prompt_short"]
      .astype(str)
      .str.replace(r"\s+", " ", regex=True)
      .str.strip()
)
dfu = dfu[dfu["prompt_short"] != ""].copy()

# 3) prompt-embeddings
prompt_vecs = ec.get_embeddings(dfu["prompt_short"].tolist())   # (N, D)

# 4) cosine-similarity
sims   = 1 - cdist(prompt_vecs, proto_vecs, metric="cosine")    # (N, K)

# 4b) hybride labels per prompt
labels_per_prompt = []
refined_ct = 0

for idx, row in enumerate(tqdm(sims, desc="Labeling (embeddings + refine)")):
    top = float(row.max())
    lbls = [i + 1 for i, s in enumerate(row) if (s >= THRESH or s == top)]

    tie_ct = int(np.isclose(row, top, rtol=0.0, atol=1e-9).sum())
    ambiguous = (BAND_LOW <= top <= BAND_HIGH) or (tie_ct > 1)

    if ambiguous:
        text = dfu.iloc[idx]["prompt_short"]
        new_lbls = refine_with_llm(text, CATEGORIES, NONE_IDX)
        if new_lbls:
            lbls = new_lbls
            refined_ct += 1

    # verwijder NONE_IDX als er andere labels zijn
    if NONE_IDX and (NONE_IDX in lbls) and (len(lbls) > 1):
        lbls = [i for i in lbls if i != NONE_IDX]

    labels_per_prompt.append(lbls)

dfu["labels"] = labels_per_prompt
print(f"ℹ️ refinement: {refined_ct} borderline prompts via {CHAT_MODEL}")

# 5) merge labels terug
df = df.merge(dfu[["prompt_short", "labels"]],
              on="prompt_short", how="left")

# 6) explode + tellen
expl = (
    df.explode("labels")
      .dropna(subset=["labels"])
      .astype({"labels": "int"})
)
if expl.empty:
    print("⚠︎ na explode geen labels; exporters krijgen ONLY 'no-choice' indien aanwezig.")

counts = (
    expl.groupby(["donor_id", "labels"])
        .size()
        .reset_index(name="n")
)
totals = expl.groupby("donor_id").size().rename("total")
counts = counts.merge(totals, on="donor_id", how="left")

# significante subtaken per donor
counts = counts[(counts["n"] >= MIN_N) & (counts["n"] / counts["total"].clip(lower=1) >= MIN_SHARE)]

# 7) lijst per donor
valid = (
    counts.groupby("donor_id")["labels"]
          .apply(list)
          .reset_index()
)

# bouw de |-gescheiden string
valid["category"] = valid["labels"].apply(
    lambda lst: "|".join(mapping[i] for i in sorted(lst))
)

# verwijder 'no-choice' als donor óók echte subtaken heeft
valid["category"] = (
    valid["category"]
      .str.replace(r"(^|[|])I did not choose .*?(?=$|[|])", "", regex=True)
      .str.replace(r"\|{2,}", "|", regex=True)
      .str.strip("|")
)
valid = valid[valid["category"] != ""].copy()
valid.drop(columns="labels", inplace=True)

# 8) donors zonder sub-labels → NONE_IDX (als aanwezig)
base_donors = set(df["donor_id"].astype(str).unique())  # na donor-filter
with_subs   = set(valid["donor_id"].astype(str))
no_subs     = sorted(base_donors - with_subs)

if no_subs and NONE_IDX:
    extra = pd.DataFrame({
        "donor_id": list(no_subs),
        "category": mapping[NONE_IDX]
    })
    valid = pd.concat([valid, extra], ignore_index=True)

# 9) metadata + export
valid = valid[["donor_id","category"]].sort_values("donor_id").reset_index(drop=True)
valid["survey_question"] = "q08_brainstorm_subtasks"
valid["timestamp"] = datetime.datetime.utcnow()

OUT_XLS.parent.mkdir(exist_ok=True, parents=True)
valid.to_excel(OUT_XLS, index=False)
print(f"✅ klaar → {OUT_XLS}")

