#!/usr/bin/env python3
"""
q06_embed.py — Task-classificatie met embeddings + (geprioriteerde) mini LLM-refinement

Doelen:
1) PER Q+A-PAIR voorspellingen (voor validatie/F1):
   results/q06_pairs.xlsx met o.a. labels, top-score en alle sim-scores.
2) Donor-samenvatting voor surveyrapportage:
   results/answers_q06.xlsx met significante categorieën.

Belangrijk:
- We labelen de user-prompts (questions), niet de antwoorden.
- GEEN harde truncation; wel payload-denoising (code/long paste/urls compacter).
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

PARSED     = pathlib.Path("parsed/all.jsonl")
PROTOS     = pathlib.Path("prototypes_q06.json")
OUT_PAIRS  = pathlib.Path("results/q06_pairs.xlsx")     # per Q+A
OUT_DONORS = pathlib.Path("results/answers_q06.xlsx")   # per donor
CACHE_NS   = "q06_tasks"

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
mapping   = {i+1: name for i, name in enumerate(CATEGORIES)}
IDX_OTHER = 6  # index van "Other"

# -----------------------------
# Thresholds / tuning
# -----------------------------
ABS_DREMPEL           = 0.28   # absolute ondergrens voor labels
REL_MARGIN            = 0.06   # relatieve marge t.o.v. top
BAND_LOW, BAND_HIGH   = 0.345, 0.355
TIE_GAP               = 0.010  # kleine top-second gap → tie
MIN_TOP_FALLBACK      = 0.26   # als niets pakt, mag top vanaf dit niveau

# ⬇️ Per Q+A-paar max 3 labels
MAX_LABELS_PER_PROMPT = None

# Donor-level significantie
MIN_N     = 5
MIN_SHARE = 0.10
# ⬇️ GEEN cap per donor (mag >3 hoofdcategorieën hebben)
TOP_K     = None

# Refinement (geprioriteerd)
REFINE_ENABLED   = True
REFINE_MAX_CALLS = 800
CHAT_MODEL       = os.getenv("CHAT_MODEL", "gpt-5-mini")
REFINE_MAXTOK    = 80

# Payload denoising (géén trunc)
MAX_EMBED_CHARS = 1200  # zachte cap na denoising

CODE_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+\-]*)\s*([\s\S]*?)```", re.MULTILINE)

def compress_payload(text: str) -> str:
    """Denoising: instructie intact; zware payload compacter (code/long paste/urls)."""
    if not isinstance(text, str):
        return ""
    s = re.sub(r"\s+", " ", text).strip()

    # 1) Code blocks → '[python block 1234 chars] <head>'
    def _repl_code(m):
        lang = (m.group(1) or "").strip().lower() or "code"
        code = m.group(2) or ""
        head = code[:300]
        return f"[{lang} block {len(code)} chars] {head}"
    s = CODE_FENCE_RE.sub(_repl_code, s)

    # 2) Lange 'paste' na triggers → maskeren
    triggers = [
        r"(?:here is the text|below is the text|the following text|tekst|text)\s*[:\-]",
        r"(?:document|article|essay)\s*[:\-]",
    ]
    s = re.sub("(" + "|".join(triggers) + r")\s+.{300,}", r"\1 [LONG_TEXT]", s, flags=re.IGNORECASE)

    # 3) URLs / e-mails korter
    s = re.sub(r"\bhttps?://\S+\b", "[URL]", s)
    s = re.sub(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "[EMAIL]", s)

    # 4) Zachte head+tail cap
    if len(s) > MAX_EMBED_CHARS:
        s = s[:MAX_EMBED_CHARS - 200] + " […] " + s[-200:]

    return s

def _extract_json_array_indices(s: str, n_max: int):
    """Pak de eerste JSON-array uit free‑form model output en filter 1..n."""
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

def refine_with_llm(prompt_text: str):
    """Vraag compacte disambiguation: alle toepasselijke categorieën (1-based)."""
    ask = (
        "Assign ALL applicable categories for the user's ChatGPT prompt from this fixed list.\n"
        "Return ONLY a JSON array of 1-based indices (e.g., [1,3]). No prose.\n\n"
        f"Categories (1..{len(CATEGORIES)}): {json.dumps(CATEGORIES, ensure_ascii=False)}\n\n"
        f"Prompt:\n{prompt_text}"
    )
    try:
        ans = gpt(ask, model=CHAT_MODEL, temperature=0, max_tokens=REFINE_MAXTOK)
    except Exception:
        return []
    return _extract_json_array_indices(str(ans), len(CATEGORIES))

# -----------------------------
# 1) prototypes + embeddings
# -----------------------------
proto_texts = json.loads(PROTOS.read_text())
assert len(proto_texts) == len(CATEGORIES), "prototypes_q06.json lengte ≠ aantal categorieën"
ec = EmbedCache(CACHE_NS)
proto_vecs  = ec.get_embeddings(proto_texts)   # (6, D)

# -----------------------------
# 2) prompts inlezen (origineel + embed-tekst)
# -----------------------------
df = pd.read_json(PARSED, lines=True)
df["embed_text"] = df["question"].apply(compress_payload)  # denoising aan
df["embed_text"] = df["embed_text"].replace("", np.nan)
df = df.dropna(subset=["embed_text"]).copy()

# unieke embed-teksten (cross-donor dedupe)
uniq = df[["embed_text"]].drop_duplicates().reset_index(drop=True)
uniq["uindex"] = np.arange(len(uniq))
df = df.merge(uniq, on="embed_text", how="left")

# -----------------------------
# 3) embedding voor unieke embed-teksten
# -----------------------------
prompt_vecs = ec.get_embeddings(uniq["embed_text"].tolist())   # (Nuniq, D)

# -----------------------------
# 4) cosine similarity (Nuniq × 6)
# -----------------------------
sims = 1 - cdist(prompt_vecs, proto_vecs, metric="cosine")    # (Nuniq, 6)

# -----------------------------
# 5) eerste pass: multi-label zonder refine
# -----------------------------
N = sims.shape[0]
top_idx_arr    = sims.argmax(axis=1) + 1
top_sim_arr    = sims.max(axis=1)
second_sim_arr = np.partition(sims, -2, axis=1)[:, -2]
gap_arr        = top_sim_arr - second_sim_arr

base_labels = [None] * N
need_refine = np.zeros(N, dtype=bool)
reason_arr  = np.array([""] * N, dtype=object)

for idx in range(N):
    row      = sims[idx]
    top_sim  = float(top_sim_arr[idx])
    top_idx  = int(top_idx_arr[idx])
    second   = float(second_sim_arr[idx])

    # multi-label: alle labels ≥ max(abs, top - margin)
    thresh = max(ABS_DREMPEL, top_sim - REL_MARGIN)
    cand = [i+1 for i, sim in enumerate(row) if sim >= thresh]

    # 'Other' alleen als er geen andere labels zijn
    if IDX_OTHER in cand and len(cand) > 1:
        cand = [c for c in cand if c != IDX_OTHER]

    # cap #labels per prompt (max 3)
    if (MAX_LABELS_PER_PROMPT is not None) and (len(cand) > MAX_LABELS_PER_PROMPT):
        cand = sorted(cand, key=lambda i: row[i-1], reverse=True)[:MAX_LABELS_PER_PROMPT]

    # fallback als leeg
    if len(cand) == 0:
        if top_idx != IDX_OTHER and top_sim >= MIN_TOP_FALLBACK:
            cand = [top_idx]
        else:
            if top_idx == IDX_OTHER and second >= MIN_TOP_FALLBACK:
                second_idx = int(np.argsort(row)[-2]) + 1
                cand = [second_idx]
            else:
                cand = [top_idx]

    base_labels[idx] = cand

    # ambigue? (alleen als er potentiële winst is)
    amb_band = (BAND_LOW <= top_sim <= BAND_HIGH)
    amb_tie  = (gap_arr[idx] <= TIE_GAP) and (top_sim >= ABS_DREMPEL)
    need_refine[idx] = (amb_band or amb_tie) and (len(cand) != 1)
    if need_refine[idx]:
        reason_arr[idx] = "band" if amb_band else "tie"

# -----------------------------
# 6) tweede pass: GEPrioriteerde refine op top-K ambigue
# -----------------------------
labels_idx_list = [list(lbls) for lbls in base_labels]  # kopie
refined_flags   = np.zeros(N, dtype=bool)

if REFINE_ENABLED and need_refine.any() and (REFINE_MAX_CALLS is not None and REFINE_MAX_CALLS > 0):
    # prioriteit: eerst borderline dicht bij band-centrum, dan kleinste gap
    band_center = (BAND_LOW + BAND_HIGH) / 2.0
    band_dist = np.where(
        (top_sim_arr >= BAND_LOW) & (top_sim_arr <= BAND_HIGH),
        np.abs(top_sim_arr - band_center),
        np.inf
    )
    pri = []
    for i in range(N):
        if not need_refine[i]:
            continue
        if reason_arr[i] == "band":
            pri.append((0, band_dist[i], i))
        else:
            pri.append((1, gap_arr[i], i))
    pri.sort()

    max_calls = min(REFINE_MAX_CALLS, len(pri))
    chosen = [i for _, __, i in pri[:max_calls]]

    calls = 0
    changed = 0
    for i in tqdm(chosen, desc="Refining borderline/tie cases"):
        text = uniq.iloc[i]["embed_text"]
        new_lbls = refine_with_llm(text)
        calls += 1
        if new_lbls:
            new_lbls = sorted({int(k) for k in new_lbls if 1 <= int(k) <= len(CATEGORIES)})
            if len(new_lbls) > 0:
                if IDX_OTHER in new_lbls and len(new_lbls) > 1:
                    new_lbls = [k for k in new_lbls if k != IDX_OTHER]
                if (MAX_LABELS_PER_PROMPT is not None) and (len(new_lbls) > MAX_LABELS_PER_PROMPT):
                    sims_i = sims[i]
                    new_lbls = sorted(new_lbls, key=lambda k: sims_i[k-1], reverse=True)[:MAX_LABELS_PER_PROMPT]
                if new_lbls != labels_idx_list[i]:
                    labels_idx_list[i] = new_lbls
                    changed += 1
                refined_flags[i] = True
    print(f"ℹ️ refinement aangeroepen: {calls}, waarvan {changed} keer labels aangepast.")
else:
    print("ℹ️ refinement overgeslagen of niet nodig.")

# -----------------------------
# 7) labels + metadata in uniq
# -----------------------------
uniq["labels_idx"]    = labels_idx_list
uniq["top_idx"]       = top_idx_arr
uniq["top_label"]     = [mapping[int(k)] for k in top_idx_arr]
uniq["top_sim"]       = top_sim_arr
uniq["refined"]       = refined_flags
uniq["refine_reason"] = reason_arr
for i in range(1, len(CATEGORIES)+1):
    uniq[f"sim_cat_{i}"] = sims[:, i-1].astype(float)

def idx_to_names(lst):
    if not isinstance(lst, (list, tuple)) or len(lst) == 0:
        return ""
    return "|".join(mapping[i] for i in sorted(set(int(x) for x in lst)))

uniq["labels"] = [idx_to_names(x) for x in labels_idx_list]

# -----------------------------
# 8) merge terug naar ALLE Q+A-paren en PER-PAIR export
# -----------------------------
pairs = df.merge(
    uniq[["uindex","labels_idx","labels","top_idx","top_label","top_sim"] +
         [f"sim_cat_{i}" for i in range(1, len(CATEGORIES)+1)] +
         ["refined","refine_reason"]],
    on="uindex", how="left"
)

# Schrijf per-pair Excel (voor validatie)
OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)
pairs_out = pairs[[
    "donor_id","conversation_id","turn_index","question",
    "labels_idx","labels","top_idx","top_label","top_sim",
    "sim_cat_1","sim_cat_2","sim_cat_3","sim_cat_4","sim_cat_5","sim_cat_6",
    "refined","refine_reason"
]].copy()
pairs_out.to_excel(OUT_PAIRS, index=False)
print(f"✅ per-Q+A predicties → {OUT_PAIRS} (rows={len(pairs_out)})")

# -----------------------------
# 9) Donor-samenvatting met significantie
# -----------------------------
expl = pairs.explode("labels_idx").dropna(subset=["labels_idx"]).astype({"labels_idx":"int"})

# counts per donor × label
counts = (
    expl.groupby(["donor_id","labels_idx"])
        .size()
        .reset_index(name="n")
)
totals = pairs.groupby("donor_id").size().rename("total")  # totaal #prompts per donor
counts = counts.merge(totals, on="donor_id")

# filter op significantie
sig = counts[(counts["n"] >= MIN_N) & (counts["n"]/counts["total"] >= MIN_SHARE)].copy()

# top-K per donor (inactief bij TOP_K=None)
if TOP_K is not None and TOP_K > 0:
    sig = (sig.sort_values(["donor_id","n"], ascending=[True, False])
              .groupby("donor_id", group_keys=False).head(TOP_K))

# labels → namen en aggregeren
if not sig.empty:
    valid = (
        sig.sort_values(["donor_id","labels_idx"])
           .groupby("donor_id")["labels_idx"]
           .apply(list)
           .reset_index()
    )
else:
    valid = pd.DataFrame(columns=["donor_id","labels_idx"])

valid["category"] = valid["labels_idx"].apply(lambda lst: "|".join(mapping[i] for i in sorted(lst)))
valid.drop(columns="labels_idx", inplace=True)

# Fallback voor donors zonder dominante categorie(ën)
all_donors  = set(pairs["donor_id"].unique())
have_labels = set(valid["donor_id"].unique())
no_labels   = sorted(all_donors - have_labels)
if no_labels:
    extra = pd.DataFrame({"donor_id": no_labels, "category": "No dominant task"})
    valid = pd.concat([valid, extra], ignore_index=True)

# metadata + export
valid["survey_question"] = "q06_tasks_used"
valid["timestamp"]       = datetime.datetime.utcnow()
OUT_DONORS.parent.mkdir(parents=True, exist_ok=True)
valid = valid.sort_values("donor_id").reset_index(drop=True)
valid.to_excel(OUT_DONORS, index=False)
print(f"✅ donor-samenvatting → {OUT_DONORS}")
