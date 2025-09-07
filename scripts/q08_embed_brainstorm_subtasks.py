#!/usr/bin/env python3
"""
q08_embed_brainstorm_subtasks.py — Brainstorming subtasks (embeddings + geprioriteerde mini LLM refine)

Q14: If you chose “Brainstorming & personal ideas / fun”, what kinds of prompts do you ask?
(multiple answers possible)
"""

import os, sys, pathlib, json, datetime, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist

# ---------- project imports ----------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from embed_cache import EmbedCache
from llm_utils import gpt   # mini LLM refinement
# -------------------------------------

PARSED     = pathlib.Path("parsed/all.jsonl")
PROTOS     = pathlib.Path("prototypes_q08.json")
Q06_XLS    = pathlib.Path("results/answers_q06.xlsx")     # bron voor Brainstorming-donors
OUT_PAIRS  = pathlib.Path("results/q08_pairs.xlsx")       # per Q+A
OUT_DONORS = pathlib.Path("results/answers_q08.xlsx")     # per donor
CACHE_NS   = "q08_brainstorm_sub"

# -----------------------------
# Tuning / drempels (gelijk aan q06)
# -----------------------------
ABS_DREMPEL       = 0.28
REL_MARGIN        = 0.06
BAND_LOW, BAND_HIGH = 0.345, 0.355
TIE_GAP           = 0.010
MIN_TOP_FALLBACK  = 0.26
MAX_LABELS_PER_PROMPT = None          # GEEN cap (Q14 multi-antwoord)

# Donor-level significantie
MIN_N     = 3
MIN_SHARE = 0.10
TOP_K     = None

# Refinement
REFINE_ENABLED   = True
REFINE_MAX_CALLS = 800
CHAT_MODEL       = os.getenv("CHAT_MODEL", "gpt-5-mini")
REFINE_MAXTOK    = 80

# Payload denoising
MAX_EMBED_CHARS = 1200
CODE_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+\-]*)\s*([\s\S]*?)```", re.MULTILINE)

def compress_payload(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = re.sub(r"\s+", " ", text).strip()
    def _repl_code(m):
        lang = (m.group(1) or "").strip().lower() or "code"
        code = m.group(2) or ""
        head = code[:300]
        return f"[{lang} block {len(code)} chars] {head}"
    s = CODE_FENCE_RE.sub(_repl_code, s)
    triggers = [
        r"(?:here is the text|below is the text|the following text|tekst|text)\s*[:\-]",
        r"(?:document|article|essay)\s*[:\-]",
    ]
    s = re.sub("(" + "|".join(triggers) + r")\s+.{300,}", r"\1 [LONG_TEXT]", s, flags=re.IGNORECASE)
    s = re.sub(r"\bhttps?://\S+\b", "[URL]", s)
    s = re.sub(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "[EMAIL]", s)
    if len(s) > MAX_EMBED_CHARS:
        s = s[:MAX_EMBED_CHARS - 200] + " […] " + s[-200:]
    return s

def _canon(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("—", "-").replace("–", "-").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s*[/\-]\s*", " - ", s)
    return re.sub(r"\s+", " ", s)

def _brainstorm_donors_from_q06() -> set[str]:
    if not Q06_XLS.exists():
        raise FileNotFoundError("results/answers_q06.xlsx niet gevonden. Run eerst q06_embed.py.")
    a6 = pd.read_excel(Q06_XLS)
    target = _canon("Brainstorming & personal ideas - fun")
    def has_brain(cat: str) -> bool:
        parts = [_canon(x) for x in str(cat).split("|")]
        return target in parts
    mask = a6["category"].astype(str).map(has_brain)
    return set(a6.loc[mask, "donor_id"].astype(str))

def _find_none_idx(categories: list[str]) -> int|None:
    for i, c in enumerate(categories, start=1):
        if "did not choose" in str(c).lower():
            return i
    return None

def _extract_json_array_indices(s: str, n_max: int) -> list[int]:
    try:
        m = re.search(r"\[[\s\S]*?\]", s);  arr = json.loads(m.group(0)) if m else []
        out = []
        for i in arr:
            ii = int(i)
            if 1 <= ii <= n_max and ii not in out: out.append(ii)
        return out
    except Exception:
        return []

def refine_with_llm(prompt_text: str, categories: list[str]) -> list[int]:
    ask = (
        "Assign ALL applicable sub-categories for the user's prompt from this fixed list.\n"
        "Return ONLY a JSON array of 1-based indices (e.g., [1,3]). No prose.\n\n"
        f"Categories (1..{len(categories)}): {json.dumps(categories, ensure_ascii=False)}\n\n"
        f"Prompt:\n{prompt_text}"
    )
    try:
        ans = gpt(ask, model=CHAT_MODEL, temperature=0, max_tokens=REFINE_MAXTOK)
    except Exception:
        return []
    return _extract_json_array_indices(str(ans), len(categories))

def _sanitize_name(name: str) -> str:
    s = re.sub(r"\s*or I did not choose.*$", "", str(name), flags=re.IGNORECASE).strip()
    return re.sub(r"[,\s]+$", "", s)

# ---------- 0) donors filter ----------
brain_donors = _brainstorm_donors_from_q06()
if not brain_donors:
    print("⚠︎ Geen donors met Brainstorming in q06. Exporteer lege sheets.")
    OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["donor_id","conversation_id","turn_index","question",
                          "labels_idx","labels","top_idx","top_label","top_sim",
                          "refined","refine_reason"]).to_excel(OUT_PAIRS, index=False)
    pd.DataFrame(columns=["donor_id","category","survey_question","timestamp"]).to_excel(OUT_DONORS, index=False)
    raise SystemExit(0)

# ---------- 1) prototypes ----------
proto_texts = json.loads(PROTOS.read_text())
CATEGORIES = [str(x) for x in proto_texts]   # 1..7 in deze volgorde
mapping = {i+1: c for i, c in enumerate(CATEGORIES)}
NONE_IDX = _find_none_idx(CATEGORIES)

ec = EmbedCache(CACHE_NS)
proto_vecs = ec.get_embeddings(CATEGORIES)   # (7, D)

# ---------- 2) prompts (alleen Brainstorming-donors) ----------
df = pd.read_json(PARSED, lines=True)
df = df[df["donor_id"].astype(str).isin(brain_donors)].copy()
df["embed_text"] = df["question"].apply(compress_payload)
df["embed_text"] = df["embed_text"].replace("", np.nan)
df = df.dropna(subset=["embed_text"]).copy()

# unieke embed-teksten (cross-donor dedupe)
uniq = df[["embed_text"]].drop_duplicates().reset_index(drop=True)
uniq["uindex"] = np.arange(len(uniq))
df = df.merge(uniq, on="embed_text", how="left")

# ---------- 3) embeddings ----------
prompt_vecs = ec.get_embeddings(uniq["embed_text"].tolist())   # (Nuniq, D)

# ---------- 4) cosine sims ----------
sims = 1 - cdist(prompt_vecs, proto_vecs, metric="cosine")     # (Nuniq, 7)
N = sims.shape[0]
top_idx_arr    = sims.argmax(axis=1) + 1
top_sim_arr    = sims.max(axis=1)
second_sim_arr = np.partition(sims, -2, axis=1)[:, -2]
gap_arr        = top_sim_arr - second_sim_arr

# ---------- 5) eerste pass (multi-label) + ambiguïteit ----------
base_labels = [None] * N
need_refine = np.zeros(N, dtype=bool)
reason_arr  = np.array([""] * N, dtype=object)

for i in range(N):
    row     = sims[i]
    top_sim = float(top_sim_arr[i])
    top_idx = int(top_idx_arr[i])
    second  = float(second_sim_arr[i])

    thresh = max(ABS_DREMPEL, top_sim - REL_MARGIN)
    cand = [k+1 for k, v in enumerate(row) if v >= thresh]

    # 'Other/Did-not-choose' niet naast andere labels
    if NONE_IDX and (NONE_IDX in cand) and len(cand) > 1:
        cand = [c for c in cand if c != NONE_IDX]

    # fallback als leeg
    if len(cand) == 0:
        if (NONE_IDX is None or top_idx != NONE_IDX) and top_sim >= MIN_TOP_FALLBACK:
            cand = [top_idx]
        else:
            if (NONE_IDX and top_idx == NONE_IDX) and second >= MIN_TOP_FALLBACK:
                second_idx = int(np.argsort(row)[-2]) + 1
                cand = [second_idx]
            else:
                cand = [top_idx]

    base_labels[i] = cand

    amb_band = (BAND_LOW <= top_sim <= BAND_HIGH)
    amb_tie  = (gap_arr[i] <= TIE_GAP) and (top_sim >= ABS_DREMPEL)
    need_refine[i] = (amb_band or amb_tie) and (len(cand) != 1)
    if need_refine[i]:
        reason_arr[i] = "band" if amb_band else "tie"

# ---------- 6) geprioriteerde refine ----------
labels_idx_list = [list(lbls) for lbls in base_labels]
refined_flags   = np.zeros(N, dtype=bool)

if REFINE_ENABLED and need_refine.any() and (REFINE_MAX_CALLS is not None and REFINE_MAX_CALLS > 0):
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

    chosen = [i for _,__,i in pri[:REFINE_MAX_CALLS]]
    calls = 0
    changed = 0
    for i in tqdm(chosen, desc="Refining borderline/tie cases"):
        text = uniq.iloc[i]["embed_text"]
        new_lbls = refine_with_llm(text, CATEGORIES)
        calls += 1
        if new_lbls:
            new_lbls = sorted({int(k) for k in new_lbls if 1 <= k <= len(CATEGORIES)})
            if NONE_IDX and (NONE_IDX in new_lbls) and len(new_lbls) > 1:
                new_lbls = [k for k in new_lbls if k != NONE_IDX]
            if len(new_lbls) == 0:
                continue
            if new_lbls != labels_idx_list[i]:
                labels_idx_list[i] = new_lbls
                changed += 1
            refined_flags[i] = True
    print(f"ℹ️ refinement aangeroepen: {calls}, waarvan {changed} keer labels aangepast.")
else:
    print("ℹ️ refinement overgeslagen of niet nodig.")

# ---------- 7) labels + metadata in uniq ----------
uniq["labels_idx"]    = labels_idx_list
uniq["top_idx"]       = top_idx_arr
uniq["top_sim"]       = top_sim_arr
uniq["refined"]       = refined_flags
uniq["refine_reason"] = reason_arr

def idx_to_names(lst):
    if not isinstance(lst, (list, tuple)) or len(lst) == 0:
        return ""
    names = [_sanitize_name(mapping[int(x)]) for x in sorted(set(int(x) for x in lst))]
    return "|".join(names)

uniq["labels"]    = [idx_to_names(x) for x in labels_idx_list]
uniq["top_label"] = [_sanitize_name(mapping[int(k)]) for k in top_idx_arr]
for i in range(1, len(CATEGORIES)+1):
    uniq[f"sim_cat_{i}"] = sims[:, i-1].astype(float)

# ---------- 8) merge → per-pair export ----------
pairs = df.merge(
    uniq[["uindex","labels_idx","labels","top_idx","top_label","top_sim"] +
         [f"sim_cat_{i}" for i in range(1, len(CATEGORIES)+1)] +
         ["refined","refine_reason"]],
    on="uindex", how="left"
)
OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)
sim_cols = [f"sim_cat_{i}" for i in range(1, len(CATEGORIES)+1)]
pairs_out = pairs[[
    "donor_id","conversation_id","turn_index","question",
    "labels_idx","labels","top_idx","top_label","top_sim",
    *sim_cols,
    "refined","refine_reason"
]].copy()
pairs_out.to_excel(OUT_PAIRS, index=False)
print(f"✅ per-Q+A predicties → {OUT_PAIRS} (rows={len(pairs_out)})")

# ---------- 9) donor-samenvatting ----------
expl = pairs.explode("labels_idx").dropna(subset=["labels_idx"]).astype({"labels_idx":"int"})
counts = expl.groupby(["donor_id","labels_idx"]).size().reset_index(name="n")
totals = pairs.groupby("donor_id").size().rename("total")
counts = counts.merge(totals, on="donor_id")

sig = counts[(counts["n"] >= MIN_N) & (counts["n"]/counts["total"] >= MIN_SHARE)].copy()

if TOP_K:
    sig = sig.sort_values(["donor_id","n"], ascending=[True, False]).groupby("donor_id", group_keys=False).head(TOP_K)

if not sig.empty:
    valid = sig.sort_values(["donor_id","labels_idx"]).groupby("donor_id")["labels_idx"].apply(list).reset_index()
else:
    valid = pd.DataFrame(columns=["donor_id","labels_idx"])

def labels_to_string(lst):
    names = [_sanitize_name(mapping[i]) for i in sorted(lst)]
    return "|".join(names)

if not valid.empty:
    valid["category"] = valid["labels_idx"].apply(labels_to_string)
    valid.drop(columns="labels_idx", inplace=True)

# fallback: donors zonder sublabels → 'Other (brainstorming-related)'
all_brain = set(pairs["donor_id"].astype(str).unique())
have_subs = set(valid["donor_id"].astype(str)) if not valid.empty else set()
no_subs   = sorted(all_brain - have_subs)
if no_subs:
    extra = pd.DataFrame({"donor_id": no_subs, "category": "Other (brainstorming-related)"})
    valid = pd.concat([valid, extra], ignore_index=True)

valid = valid.sort_values("donor_id").reset_index(drop=True)
valid["survey_question"] = "q08_brainstorm_subtasks"
valid["timestamp"] = datetime.datetime.utcnow()
OUT_DONORS.parent.mkdir(parents=True, exist_ok=True)
valid.to_excel(OUT_DONORS, index=False)
print(f"✅ donor-samenvatting → {OUT_DONORS}")
