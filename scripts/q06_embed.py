#!/usr/bin/env python3
"""
q06_embed.py — hoofdcategorieën met embeddings + mini LLM-refinement

Doel:
- Alleen significante categorieën per donor meenemen.
- Significantie: minimaal MIN_N hits én MIN_SHARE van alle prompts van die donor.
- (Optioneel) Beperk tot maximaal TOP_K categorieën per donor (op basis van counts).

Output: results/answers_q06.xlsx met kolommen:
- donor_id, category (|-gescheiden), survey_question="q06_tasks_used", timestamp
"""
import os, pathlib, json, datetime, sys, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist

# project imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from embed_cache import EmbedCache
from llm_utils import gpt

PARSED   = pathlib.Path("parsed/all.jsonl")
PROTOS   = pathlib.Path("prototypes_q06.json")
OUT_XLS  = pathlib.Path("results/answers_q06.xlsx")
CACHE_NS = "q06_tasks"

# -----------------------------
# Categorieën (order is 1..6)
# -----------------------------
CATEGORIES = [
    "Writing & professional communication",
    "Brainstorming & personal ideas - fun",
    "Coding - programming help",
    "Language practice or translation",
    "Study revision - exam prep",
    "Other",
]
mapping = {i+1: name for i, name in enumerate(CATEGORIES)}

# -----------------------------
# Thresholds
# -----------------------------
THRESH    = 0.35          # per-prompt label-drempel
BAND_LOW  = 0.345         # borderline band voor mini-refine
BAND_HIGH = 0.355

# Significantie-drempels per donor (streng)
MIN_N     = 5             # min. aantal prompts voor een categorie
MIN_SHARE = 0.10          # min. aandeel binnen donor
TOP_K     = 3             # max. aantal categorieën per donor (None om uit te zetten)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-mini")

# -----------------------------
# helpers
# -----------------------------
def _extract_json_array_indices(s: str, n_max: int) -> list[int]:
    """Zoek de eerste JSON array in tekst en filter op geldige 1..n indices."""
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

def refine_with_llm(prompt_text: str) -> list[int]:
    """Vraag gpt-5-mini om ALLE toepasselijke categorieën te geven (1-based)."""
    ask = (
        "Assign ALL applicable categories for the user's ChatGPT prompt from this fixed list.\n"
        "Return ONLY a JSON array of 1-based indices (e.g., [1,3]). No prose.\n\n"
        f"Categories (1..{len(CATEGORIES)}): {json.dumps(CATEGORIES, ensure_ascii=False)}\n\n"
        f"Prompt:\n{prompt_text}"
    )
    try:
        ans = gpt(ask, model=CHAT_MODEL, temperature=0, max_tokens=80)
    except Exception:
        return []
    return _extract_json_array_indices(str(ans), len(CATEGORIES))

# -----------------------------
# 1) prototypes embedden
# -----------------------------
proto_texts = json.loads(PROTOS.read_text())
ec = EmbedCache(CACHE_NS)
proto_vecs  = ec.get_embeddings(proto_texts)   # (6, D)

# -----------------------------
# 2) unieke prompts inlezen + clean
# -----------------------------
df = pd.read_json(PARSED, lines=True)
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

# -----------------------------
# 3) embeddings prompts
# -----------------------------
prompt_vecs = ec.get_embeddings(dfu["prompt_short"].tolist())   # (N, D)

# -----------------------------
# 4) cosine similarity N×6
# -----------------------------
sims = 1 - cdist(prompt_vecs, proto_vecs, metric="cosine")      # (N, 6)

# -----------------------------
# 4b) labels + hybride refine
# -----------------------------
labels_per_prompt = []
refined_ct = 0

for idx, row in enumerate(tqdm(sims, desc="Labeling (embeddings + refine)")):
    top = float(row.max())
    # basislabels: >= THRESH of de top
    lbls = [i+1 for i, sim in enumerate(row) if (sim >= THRESH or sim == top)]

    # twijfel: score in band of meerdere labels exact/top gelijk
    tie_ct = int(np.isclose(row, top, rtol=0.0, atol=1e-9).sum())
    ambiguous = (BAND_LOW <= top <= BAND_HIGH) or (tie_ct > 1)

    if ambiguous:
        text = dfu.iloc[idx]["prompt_short"]
        new_lbls = refine_with_llm(text)
        if new_lbls:      # alleen vervangen bij geldige respons
            lbls = new_lbls
            refined_ct += 1

    labels_per_prompt.append(lbls)

dfu["labels"] = labels_per_prompt
print(f"ℹ️ refinement: {refined_ct} borderline prompts via {CHAT_MODEL}")

# -----------------------------
# 5) merge labels terug op hoofd-df
# -----------------------------
df = df.merge(dfu[["prompt_short", "labels"]], on="prompt_short", how="left")

# -----------------------------
# 6) explode en significante categorieën selecteren
# -----------------------------
expl = (
    df.explode("labels")
      .dropna(subset=["labels"])
      .astype({"labels": "int"})
)

# counts per donor × label
counts = (
    expl.groupby(["donor_id", "labels"])
        .size()
        .reset_index(name="n")
)
# totalen per donor
totals = expl.groupby("donor_id").size().rename("total")
counts = counts.merge(totals, on="donor_id")

# strenge significantie-filter
sig = counts[(counts["n"] >= MIN_N) & (counts["n"] / counts["total"] >= MIN_SHARE)].copy()

# (optioneel) Beperk tot top-K labels per donor op basis van 'n'
if TOP_K is not None and TOP_K > 0:
    sig = sig.sort_values(["donor_id", "n"], ascending=[True, False])
    sig = sig.groupby("donor_id", group_keys=False).head(TOP_K)

# bouw lijst per donor
if not sig.empty:
    valid = (
        sig.sort_values(["donor_id", "labels"])
           .groupby("donor_id")["labels"]
           .apply(list)
           .reset_index()
    )
else:
    valid = pd.DataFrame(columns=["donor_id","labels"])

# labels → namen
valid["category"] = valid["labels"].apply(
    lambda lst: "|".join(mapping[i] for i in sorted(lst))
)
valid.drop(columns="labels", inplace=True)

# -----------------------------
# 6b) Fallback: donors zonder dominante categorie → 'Other (no dominant task)'
# -----------------------------
all_donors  = set(df["donor_id"].unique())
have_labels = set(valid["donor_id"].unique())
no_labels   = sorted(all_donors - have_labels)

if no_labels:
    print(f"ℹ️ fallback voor {len(no_labels)} donor(s) zonder dominante categorie: {no_labels}")
    extra = pd.DataFrame({
        "donor_id": no_labels,
        "category": "Other (no dominant task)"
    })
    valid = pd.concat([valid, extra], ignore_index=True)

# -----------------------------
# 7) metadata + export
# -----------------------------
valid["survey_question"] = "q06_tasks_used"
valid["timestamp"] = datetime.datetime.utcnow()

OUT_XLS.parent.mkdir(exist_ok=True, parents=True)
valid = valid.sort_values("donor_id").reset_index(drop=True)
valid.to_excel(OUT_XLS, index=False)
print(f"✅ klaar → {OUT_XLS}")

