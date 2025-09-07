# -*- coding: utf-8 -*-
"""
02c_build_llogs_subtasks_from_prompts.py
Creates Q13–Q17 subtask one-hots for Llogs from prompt-level logs using regex rules.
- Auto-detects a prompt-level file (parquet/csv) in results/derived/ or logs/
- Multi-label matching per prompt
- Donor-level thresholds: n>=3 and share>=10% of family prompts (per H4)
- Gated on Q12 parent families present in Llogs_clean.csv
Outputs:
  - results/derived/Llogs_clean.csv  (updated with subtask one-hots)
  - results/derived/Llogs_subtasks_counts.csv
  - results/derived/Llogs_promptlevel_subtasks_sample.csv
"""

import os, re, json, ast
import numpy as np
import pandas as pd

BASE = os.getcwd()
DERIVED = os.path.join(BASE, "results", "derived")
LOGS_DIRS = [DERIVED, os.path.join(BASE, "logs"), os.path.join(BASE, "logs", "parsed"), os.path.join(BASE, "data"), os.path.join(BASE, "data", "parsed")]
os.makedirs(DERIVED, exist_ok=True)

# ------------------ helpers ------------------ #
def find_prompt_file():
    candidates = []
    for d in LOGS_DIRS:
        if not os.path.isdir(d): continue
        for fn in os.listdir(d):
            low = fn.lower()
            if (low.endswith(".parquet") or low.endswith(".csv")) and any(tok in low for tok in ["log", "llog", "chat", "qa", "prompt"]):
                candidates.append(os.path.join(d, fn))
    # prefer parquet in derived
    prefs = sorted(candidates, key=lambda p: (not p.endswith(".parquet"), "derived" not in p.lower()))
    return prefs[0] if prefs else None

def load_prompt_table(path):
    if path.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print(f"[warn] parquet failed: {e}; trying csv fallback")
    return pd.read_csv(path)

def pick_first_column(df, names):
    for n in names:
        if n in df.columns: return n
    # try case-insensitive
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low: return low[n.lower()]
    return None

def norm_text(x):
    if pd.isna(x): return ""
    return str(x).strip()

def any_re(txt, patterns):
    t = txt.lower()
    return any(re.search(p, t) for p in patterns)

def dedup_keep_first(seq):
    seen = set(); out = []
    for x in seq:
        if x in seen: continue
        seen.add(x); out.append(x)
    return out

# ------------------ regex codebook ------------------ #
# Minimal, auditable keyword sets; expand safely if needed
CB = {
    "Q13": {  # Writing subtasks
        "outlining_ideas_or_slides": [
            r"\boutline", r"\bbullet points?\b", r"\bstructure\b", r"\bheadings?\b", r"\bslides?\b", r"\bdeck\b"
        ],
        "drafting_full_text": [
            r"\bdraft\b", r"\bwrite\b", r"\bcompose\b", r"\bessay\b", r"\bemail\b", r"\breport\b", r"\bcover letter\b", r"\barticle\b"
        ],
        "proof_reading_tone_adjustment": [
            r"\bproof ?read", r"\bgrammar\b", r"\btypos?\b", r"\bpolish\b", r"\bcopy ?edit", r"\btone\b", r"\bmake it (more|less)\b"
        ],
        "summarising_sources_or_meeting_notes": [
            r"\bsummar(y|ise|ize)\b", r"\btl;?dr\b", r"\bbrief\b", r"\bcondens", r"\bmeeting notes?\b"
        ],
        "adjusting_style_for_different_audiences": [
            r"\brewrite for\b", r"\bsimplif(y|ied) for\b", r"\bmake it (formal|casual|professional|concise)\b", r"\bfor (kids|children|beginners|lay(wo)?men)\b"
        ],
    },
    "Q14": {  # Brainstorming/fun
        "academic_or_research_topics": [
            r"\bresearch\b", r"\bthesis\b", r"\bresearch questions?\b", r"\bliterature\b", r"\bhypothesi?s\b", r"\bpaper topic"
        ],
        "creative_roleplay_jokes_stories": [
            r"\brole ?play\b", r"\bjokes?\b", r"\bstor(y|ies)\b", r"\bpoem\b", r"\bsong\b", r"\brap\b", r"\bcharacter\b", r"\bd&d\b", r"\bfanfic"
        ],
        "what_if_scenarios": [
            r"\bwhat if\b", r"\bcounterfactual", r"\bimagine if\b", r"\bscenario (where|in which)\b"
        ],
        "recommendations_books_movies_music": [
            r"\brecommend", r"\bsuggest", r"\bbooks?\b", r"\bmovies?\b", r"\bfilms?\b", r"\bmusic\b", r"\bplaylist\b", r"\bseries\b", r"\bpodcasts?\b"
        ],
        "trivia_and_general_knowledge": [
            r"\bwho is\b", r"\bwhat is\b", r"\bwhen did\b", r"\bcapital of\b", r"\bhow many\b"
        ],
    },
    "Q15": {  # Coding
        "generating_new_code_snippets": [
            r"\bwrite .*code", r"\bimplement\b", r"\bfunction that\b", r"\bscript to\b", r"\bgenerate code\b", r"\balgorithm\b", r"\bbuilder?\b"
        ],
        "debugging_existing_code": [
            r"\bbug\b", r"\berror\b", r"\btraceback\b", r"\bfix (this|my)\b", r"\bdoesn'?t work\b", r"\bdebug\b", r"\bexception\b"
        ],
        "explaining_code_or_concepts": [
            r"\bexplain\b.*\b(code|algorithm|complexit(y|ies)|regex|dsl)\b", r"\bwhat does this code do\b", r"\bhow (does|do) .* work"
        ],
        "converting_code_between_languages": [
            r"\bconvert\b.*\bto\b.*\bfrom\b|\bconvert .* from .* to\b", r"\btranslate\b.*\bcode\b", r"\bport\b.*\bto\b"
        ],
        "writing_unit_tests": [
            r"\bunit tests?\b", r"\bpytest\b", r"\btest cases?\b", r"\bassert\b", r"\bcoverage\b"
        ],
    },
    "Q16": {  # Language/translation
        "translating_full_texts_between_languages": [
            r"\btranslate\b.*\bto\b", r"\btranslation\b", r"\bfrom\b.*\bto\b"
        ],
        "improving_grammar_or_style_in_target_language": [
            r"\bcorrect my grammar\b", r"\bfix grammar\b", r"\bimprove .* (writing|text)\b", r"\bpolish\b", r"\bproof ?read\b"
        ],
        "vocabulary_drills_or_word_lists": [
            r"\bvocabulary\b", r"\bword list\b", r"\bwordlist\b", r"\bsynonyms?\b", r"\bphrases?\b"
        ],
        "conversational_practice_or_dialogue_roleplay": [
            r"\bpractice\b.*\bconversation\b", r"\bchat with me in\b", r"\brole ?play\b.*(teacher|tutor|native speaker)"
        ],
        "pronunciation_or_phonetic_guidance": [
            r"\bpronunciation\b", r"\bhow to pronounce\b", r"\bIPA\b", r"\bphonetic"
        ],
    },
    "Q17": {  # Study/exam
        "summarising_lecture_notes_or_readings": [
            r"\bsummar(y|ise|ize)\b.*(lecture|notes?|paper|article|reading)", r"\blecture notes?\b", r"\breadings?\b"
        ],
        "generating_practice_questions_or_quizzes": [
            r"\bquiz\b", r"\bpractice questions?\b", r"\btest me\b", r"\bmultiple[- ]choice\b", r"\bmcq\b"
        ],
        "explaining_difficult_concepts_in_simple_terms": [
            r"\bexplain\b.*\bconcept\b|\bELI5\b|\bsimplif(y|ied)\b|\bteach me\b"
        ],
        "reviewing_flashcards_or_key_terms": [
            r"\bflashcards?\b", r"\banki\b", r"\bkey terms?\b", r"\bdefinitions?\b"
        ],
    }
}

# --------------- load donor-level base --------------- #
LL_PATH = os.path.join(DERIVED, "Llogs_clean.csv")
if not os.path.exists(LL_PATH):
    raise FileNotFoundError(f"Missing {LL_PATH}. Run your cleaning step first.")

Llogs = pd.read_csv(LL_PATH, dtype=str, keep_default_na=False)

# Expect a donor id column in Llogs_clean
DONOR_KEYS = ["donor_id", "donor", "participant_id", "anon_id", "id"]
DONOR_COL = next((c for c in DONOR_KEYS if c in Llogs.columns), None)
if DONOR_COL is None:
    raise RuntimeError("Could not find a donor id column in Llogs_clean.csv (looked for donor_id/donor/participant_id/anon_id/id).")

# --------------- load prompt-level logs --------------- #
prompt_path = find_prompt_file()
if not prompt_path:
    raise FileNotFoundError("Could not auto-detect a prompt-level logs file in results/derived/ or logs/. "
                            "Place a parquet/csv with prompts there (e.g., L_logs.parquet).")

print(f"[info] Using prompt-level file: {prompt_path}")
PL = load_prompt_table(prompt_path)

# identify donor id & prompt text columns
PL_DONOR = pick_first_column(PL, [DONOR_COL, "donor_id", "donor", "participant_id", "anon_id", "id", "user"])
PL_TEXT  = pick_first_column(PL, ["question_text", "prompt", "prompt_text", "user_text", "content", "message", "input_text", "input"])
if PL_DONOR is None or PL_TEXT is None:
    raise RuntimeError(f"Could not find donor and prompt text columns. Found donor={PL_DONOR} prompt={PL_TEXT}. "
                       f"Columns available: {list(PL.columns)[:30]} ...")

PL = PL[[PL_DONOR, PL_TEXT]].rename(columns={PL_DONOR: "donor_id", PL_TEXT: "prompt_text"})
PL["donor_id"] = PL["donor_id"].astype(str)
PL["prompt_text"] = PL["prompt_text"].map(norm_text)

# --------------- per-prompt matching --------------- #
def match_prompt(prompt):
    fam_hits = {}
    for fam, subtasks in CB.items():
        hits = []
        for st, pats in subtasks.items():
            if any_re(prompt, pats):
                hits.append(st)
        fam_hits[fam] = dedup_keep_first(hits)
    return fam_hits

# apply matcher (vectorized-ish)
matched = PL["prompt_text"].apply(match_prompt)
# explode to long: each row per prompt per family with list of subtasks
rows = []
for (donor, txt), famdict in zip(PL[["donor_id","prompt_text"]].itertuples(index=False, name=None), matched):
    # famdict is a dict (Q13..Q17: [subtasks])
    for fam, subs in famdict.items():
        # record family presence if any subtask matched
        if subs:
            rows.append({"donor_id": donor, "family": fam, "prompt_id": id(txt), "subtask": "__ANY__"})  # family presence
            for st in subs:
                rows.append({"donor_id": donor, "family": fam, "prompt_id": id(txt), "subtask": st})

ML = pd.DataFrame(rows)
if ML.empty:
    print("[warn] No prompts matched any subtask patterns. The regex codebook may be too strict.")
    # still proceed with empty outputs below
else:
    # family totals = #unique prompts with any subtask in family
    fam_tot = (ML[ML["subtask"]=="__ANY__"]
               .drop_duplicates(["donor_id","family","prompt_id"])
               .groupby(["donor_id","family"]).size()
               .rename("family_total").reset_index())

    # subtask counts = #unique prompts with that specific subtask
    st_cnt = (ML[ML["subtask"]!="__ANY__"]
              .drop_duplicates(["donor_id","family","prompt_id","subtask"])
              .groupby(["donor_id","family","subtask"]).size()
              .rename("count").reset_index())

    # combine
    agg = st_cnt.merge(fam_tot, on=["donor_id","family"], how="left")
    agg["share"] = np.where(agg["family_total"]>0, agg["count"]/agg["family_total"], np.nan)

    # thresholds
    agg["passes_thresholds"] = (agg["count"]>=3) & (agg["share"]>=0.10)

    # map families to Q12 parents (for gating)
    PARENT_COL = {
        "Q13": "q12__writing_and_professional_communication",
        "Q14": "q12__brainstorming_and_personal_ideas_fun",
        "Q15": "q12__coding_programming_help",
        "Q16": "q12__language_practice_or_translation",
        "Q17": "q12__study_revision_or_exam_prep",
    }

    # merge parent flags from Llogs_clean
    parents = Llogs[[DONOR_COL] + list(PARENT_COL.values())].copy()
    parents = parents.rename(columns={DONOR_COL: "donor_id"})
    for fam, parent in PARENT_COL.items():
        if parent not in parents.columns:
            # if missing, create 0 (then nothing will pass gating for that family)
            parents[parent] = 0
    # long format parents
    par_long = (parents.melt(id_vars=["donor_id"], var_name="parent_col", value_name="parent_flag"))
    inv = {v:k for k,v in PARENT_COL.items()}
    par_long["family"] = par_long["parent_col"].map(inv)
    par_long["parent_flag"] = pd.to_numeric(par_long["parent_flag"], errors="coerce").fillna(0).astype(int)

    agg = agg.merge(par_long[["donor_id","family","parent_flag"]], on=["donor_id","family"], how="left")
    agg["gated_pass"] = agg["passes_thresholds"] & (agg["parent_flag"]==1)

    # write debug counts
    agg.to_csv(os.path.join(DERIVED, "Llogs_subtasks_counts.csv"), index=False)

    # build wide donor-level one-hots per subtask (names aligned to qXX__subtask)
    wide = (agg[agg["gated_pass"]]
            .assign(subtask_col=lambda d: d["family"].str.lower() + "__" + d["subtask"])
            .pivot_table(index="donor_id", columns="subtask_col", values="gated_pass", aggfunc="max")
            .fillna(0).astype(int).reset_index())

    # merge into Llogs_clean on donor id
    Llogs_merged = Llogs.copy()
    Llogs_merged = Llogs_merged.rename(columns={DONOR_COL: "donor_id"})
    # remove any pre-existing q13..q17 one-hots to avoid duplicates
    pre_cols = [c for c in Llogs_merged.columns if re.match(r"^q1[3-7][^A-Za-z0-9]+", c)]
    if pre_cols:
        Llogs_merged = Llogs_merged.drop(columns=pre_cols)

    Llogs_merged = Llogs_merged.merge(wide, on="donor_id", how="left")
    # fill missing new one-hots with 0
    new_cols = [c for c in Llogs_merged.columns if re.match(r"^q1[3-7][^A-Za-z0-9]+", c)]
    for c in new_cols:
        Llogs_merged[c] = pd.to_numeric(Llogs_merged[c], errors="coerce").fillna(0).astype(int)

    # restore original donor id column name
    Llogs_merged = Llogs_merged.rename(columns={"donor_id": DONOR_COL})

    Llogs_merged.to_csv(LL_PATH, index=False)
    print(f"[ok] Updated {LL_PATH} with {len(new_cols)} subtask one-hot columns.")

    # sample for manual QA
    sample_rows = min(200, len(PL))
    (PL.head(sample_rows)
       .to_csv(os.path.join(DERIVED, "Llogs_promptlevel_subtasks_sample.csv"), index=False))

    print("[done] Wrote counts table and sample to 'results/derived/'.")

