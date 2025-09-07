# -*- coding: utf-8 -*-
"""
00_rebuild_sdon_clean.py — Rebuild Sdon_clean.csv from the latest raw "Survey Donor (24).csv"

Outputs:
  results/derived/Sdon_clean.csv

Same logic as 00_rebuild_llogs_clean.py:
- robust Q7–Q11
- Q12 families
- Q13–Q17 subtasks (multi-select OR one-hot); subtask breadth per family
"""

import os, re, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

BASE = os.getcwd()
RAW_CANDIDATES = [
    os.path.join(BASE, "data", "raw", "Survey Donor (24).csv"),
    os.path.join(BASE, "data", "raw", "Survey donor (24).csv"),
    os.path.join(BASE, "Survey Donor (24).csv"),
    os.path.join(BASE, "Survey donor (24).csv"),
]
OUT_DIR = os.path.join(BASE, "results", "derived")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helpers (same as Llogs) ----------
def find_file(candidates):
    for p in candidates:
        if os.path.exists(p): return p
    raise FileNotFoundError("Could not find the raw Sdon CSV.\n  - " + "\n  - ".join(candidates))

def read_csv_any(path): return pd.read_csv(path, dtype=str, keep_default_na=False)
def strip_all(df): return df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

def coerce01(s):
    s = s.astype(str).str.strip().str.lower()
    m1 = s.isin(["1","true","yes","y","ja"]); m0 = s.isin(["0","false","no","n","nee",""])
    out = pd.Series(np.nan, index=s.index, dtype=float); out[m1]=1; out[m0]=0
    out = out.fillna(pd.to_numeric(s, errors="coerce"))
    return out.fillna(0).astype(int)

def map_q7_band(x):
    s = str(x).lower().strip()
    if s in {"0","none"}: return 0.0
    if "1" in s and "2" in s: return 1.5
    if "3" in s and "5" in s: return 4.0
    if "6" in s and "10" in s: return 8.0
    if ">" in s or "more than 10" in s: return 12.0
    try: return float(s)
    except: return np.nan

def map_q8_band(x):
    s = str(x).lower().strip().replace(" ", "")
    if s == "0": return 0.0
    if s == "1": return 1.0
    if "2-3" in s or "2–3" in s: return 2.5
    if "4-5" in s or "4–5" in s: return 4.5
    if "6+" in s or "6plus" in s or "morethan6" in s: return 6.0
    try: return float(str(x))
    except: return np.nan

def map_q9_band(x):
    s = str(x).lower().strip().replace(" ", "")
    if s in {"<5","<05"}: return 2.5
    if "5-15" in s or "05-15" in s: return 10.0
    if "15-30" in s: return 22.5
    if "30-60" in s: return 45.0
    if ">60" in s or "morethan60" in s: return 75.0
    try: return float(str(x))
    except: return np.nan

def canonical_q10(x):
    s = str(x).strip().lower()
    if "even" in s: return "Evenings"
    if "work" in s or "study" in s: return "During work / study hours"
    if "any" in s: return "Anytime throughout the day"
    return str(x).strip() if str(x).strip() else "Anytime throughout the day"

def canonical_q11_band(x):
    s = str(x).strip().lower()
    if "multiple" in s: return "multiple_paragraphs"
    if "short" in s and "paragraph" in s: return "short_paragraph"
    if "short" in s and "sentence" in s: return "short_sentence"
    if "var" in s: return "varies"
    if s in {"multiple_paragraphs","short_paragraph","short_sentence","varies"}: return s
    return "varies" if s=="" else s

def q11_score(code): return {"short_sentence":1,"short_paragraph":2,"multiple_paragraphs":3}.get(code, np.nan)

Q12_FAMS = {
    "q12__writing_and_professional_communication": ["writing & communication","writing and communication","writing","communication","email","report","wri"],
    "q12__brainstorming_and_personal_ideas_fun": ["brainstorming / fun","brainstorming and personal ideas fun","brainstorming","ideas","fun","bra"],
    "q12__coding_programming_help": ["coding / programming","coding programming help","coding","programming","code","cod"],
    "q12__language_practice_or_translation": ["language / translation","language practice or translation","language","translation","lan"],
    "q12__study_revision_or_exam_prep": ["study / exam","study revision or exam prep","study","exam","revision","stu"],
    "q12__other": ["other","oth"]
}
Q12_COLS = list(Q12_FAMS.keys())

def detect_families_from_text(cell: str):
    hit = {k:0 for k in Q12_COLS}
    s = str(cell).strip()
    if not s: return hit
    parts = [p.strip().lower() for p in re.split(r"[;,]", s) if p.strip()]
    for p in parts:
        if "did not choose" in p: continue
        for fam, keys in Q12_FAMS.items():
            if any(key in p for key in keys): hit[fam] = 1
    return hit

SUBTASKS = {
    "q13": {
        "outlining_ideas_or_slides": ["outline","slide","slides","bullet","structure","headings","deck"],
        "drafting_full_text": ["draft","write","full text","email","essay","report","compose"],
        "proof_reading_tone_adjustment": ["proof","grammar","tone","polish","edit","copyedit","proofread"],
        "summarising_sources_or_meeting_notes": ["summar","tl;dr","brief","condens","meeting notes","sources","notes"],
        "adjusting_style_for_different_audiences": ["style","audience","formal","casual","professional","kids","beginners","simplif","rewrite"],
        "other": ["other"]
    },
    "q14": {
        "academic_or_research_topics": ["academic","research","thesis","paper","hypothesis","literature","topic"],
        "creative_role_play_jokes_stories": ["role play","roleplay","joke","story","poem","song","rap","character","creative"],
        "what_if_scenarios": ["what if","hypothetical","scenario","counterfactual"],
        "recommendations_books_movies_music": ["recommend","suggest","book","movie","film","music","playlist","series","podcast"],
        "trivia_and_general_knowledge": ["trivia","general knowledge","facts","quiz"],
        "other": ["other"]
    },
    "q15": {
        "generating_new_code_snippets": ["generate code","write code","new code","implement","function","script","algorithm"],
        "debugging_existing_code": ["debug","bug","error","fix","traceback","exception"],
        "explaining_code_or_concepts": ["explain code","explain","concept","what does this code do","how .* work","complexit","regex"],
        "converting_code_between_languages": ["convert","translate code","port","from .* to","to .* from"],
        "writing_unit_tests": ["unit test","pytest","test case","assert","coverage"],
        "other": ["other"]
    },
    "q16": {
        "translating_full_texts_between_languages": ["translate","translation","from .* to","to .* from"],
        "improving_grammar_or_style_in_a_target_language": ["grammar","style","proofread","polish","improve"],
        "vocabulary_drills_or_word_lists": ["vocabulary","word list","synonym","phrases"],
        "conversational_practice_or_dialogue_role_play": ["conversation","chat","role play","roleplay","dialogue"],
        "pronunciation_or_phonetic_guidance": ["pronunciation","pronounce","ipa","phonetic"],
        "other": ["other"]
    },
    "q17": {
        "summarising_lecture_notes_or_readings": ["summary","summar","lecture","notes","reading","paper","article"],
        "generating_practice_questions_or_quizzes": ["quiz","practice questions","test me","multiple choice","mcq"],
        "explaining_difficult_concepts_in_simple_terms": ["explain","concept","eli5","simplif","teach me"],
        "reviewing_flashcards_or_key_terms": ["flashcard","anki","key terms","definitions"],
        "other": ["other"]
    }
}
SUB_BREADTH_KEYS = {"q13":"subtask_breadth_wri","q14":"subtask_breadth_bra","q15":"subtask_breadth_cod","q16":"subtask_breadth_lan","q17":"subtask_breadth_stu"}

def norm(s): return re.sub(r"\s+"," ", str(s).strip().lower())

def detect_subtasks_from_text(cell, qtag):
    d = {f"{qtag}__{k}":0 for k in SUBTASKS[qtag].keys()}
    s = str(cell).strip()
    if not s: return d
    parts = [norm(p) for p in re.split(r"[;,]", s) if p.strip()]
    for p in parts:
        if "did not choose" in p: continue
        for canon, keys in SUBTASKS[qtag].items():
            if any(k in p for k in keys): d[f"{qtag}__{canon}"] = 1
    return d

def detect_subtasks_from_columns(df, qtag):
    cols = [c for c in df.columns if re.match(rf"^{qtag}\b", c, flags=re.IGNORECASE)]
    out = pd.DataFrame(index=df.index)
    for canon, keys in SUBTASKS[qtag].items():
        m = np.zeros(len(df), dtype=int)
        for c in cols:
            low = norm(c)
            if "not_chosen" in low or "not chosen" in low or "breadth" in low: continue
            if any(k in low for k in keys): m = np.maximum(m, coerce01(df[c]).values)
        out[f"{qtag}__{canon}"] = m
    for canon in SUBTASKS[qtag].keys():
        col = f"{qtag}__{canon}"
        if col not in out.columns: out[col] = 0
    return out

def derive_subtasks(df, qtag):
    text_candidates = [c for c in df.columns if re.match(rf"^{qtag}\b", c, flags=re.IGNORECASE)]
    text_col = None
    for c in text_candidates:
        vals = df[c].astype(str)
        if vals.str.contains(r"[;,]").any() or vals.str.len().median() > 15:
            text_col = c; break
    if text_col is not None and len(text_candidates) <= 2:
        D = df[text_col].apply(lambda x: detect_subtasks_from_text(x, qtag))
        out = pd.DataFrame(list(D.values), index=df.index)
    else:
        out = detect_subtasks_from_columns(df, qtag)
    canon_cols = [c for c in out.columns if c.startswith(f"{qtag}__")]
    excl = [c for c in canon_cols if c.endswith("__other")]
    breadth = out[ [c for c in canon_cols if c not in excl] ].sum(axis=1)
    out[SUB_BREADTH_KEYS[qtag]] = breadth.astype(int)
    return out

# ---------- pipeline ----------
raw_path = find_file(RAW_CANDIDATES)
df = read_csv_any(raw_path)
df = strip_all(df)

# Drop "Response ID" first row if present
first_row = df.iloc[0].astype(str).str.lower().tolist()
if any("response id" in v for v in first_row):
    df = df.iloc[1:].reset_index(drop=True)

# Harmonise names
df.columns = [re.sub(r"\s+","_", c.strip().lower()) for c in df.columns]

out = pd.DataFrame(index=df.index.copy())

# IDs
for idcand in ["responseid","response_id","donor_id","id"]:
    if idcand in df.columns:
        out["donor_id"] = df[idcand]
        break

# Q7–Q9
out["Q7_mid"] = pd.to_numeric(df.get("q7_mid", pd.Series(np.nan, index=df.index)), errors="coerce")
if out["Q7_mid"].isna().all():
    cand = "q7" if "q7" in df.columns else ("q7_band" if "q7_band" in df.columns else None)
    if cand is None: raise KeyError("Missing Q7")
    out["Q7_mid"] = df[cand].map(map_q7_band)

out["Q8_mid"] = pd.to_numeric(df.get("q8_mid", pd.Series(np.nan, index=df.index)), errors="coerce")
if out["Q8_mid"].isna().all():
    cand = "q8" if "q8" in df.columns else ("q8_band" if "q8_band" in df.columns else None)
    if cand is None: raise KeyError("Missing Q8")
    out["Q8_mid"] = df[cand].map(map_q8_band)

out["Q9_mid"] = pd.to_numeric(df.get("q9_mid", pd.Series(np.nan, index=df.index)), errors="coerce")
if out["Q9_mid"].isna().all():
    cand = "q9" if "q9" in df.columns else ("q9_band" if "q9_band" in df.columns else None)
    if cand is None: raise KeyError("Missing Q9")
    out["Q9_mid"] = df[cand].map(map_q9_band)

# Q10/Q11
out["Q10"] = df.get("q10", df.get("daypart","Anytime throughout the day")).map(canonical_q10)
q11_src = None
for cand in ["q11_band","q11","prompt_length","prompt_length_band"]:
    if cand in df.columns: q11_src = cand; break
if q11_src is None:
    out["Q11_band"]="varies"; out["Q11_score"]=np.nan
else:
    out["Q11_band"]=df[q11_src].map(canonical_q11_band)
    out["Q11_score"]=out["Q11_band"].map(q11_score)

# Q12 families
present_binaries = [c for c in Q12_COLS if c in df.columns]
if len(present_binaries) == len(Q12_COLS):
    for c in Q12_COLS: out[c] = coerce01(df[c])
else:
    text_col = None
    for cand in ["q12","q12_families","families_q12","tasks_q12"]:
        if cand in df.columns: text_col = cand; break
    if text_col is None:
        for c in Q12_COLS: out[c] = coerce01(df[c]) if c in df.columns else 0
    else:
        hits = df[text_col].apply(detect_families_from_text)
        for c in Q12_COLS: out[c] = hits.apply(lambda d: d.get(c,0)).astype(int)
out["task_breadth_main"] = out[Q12_COLS].sum(axis=1)

# Q13–Q17 subtasks
for qtag in ["q13","q14","q15","q16","q17"]:
    sub_df = derive_subtasks(df, qtag)
    out = pd.concat([out, sub_df], axis=1)

# Final ordering
sub_cols = [c for c in out.columns if re.match(r"^q1[3-7]__", c)]
final_cols = ["donor_id","Q7_mid","Q8_mid","Q9_mid","Q10","Q11_band","Q11_score","task_breadth_main"] + Q12_COLS + sub_cols + list(SUB_BREADTH_KEYS.values())
final_cols = [c for c in final_cols if c in out.columns]
out = out[final_cols].copy()

# Save
csv_path = os.path.join(OUT_DIR, "Sdon_clean.csv")
out.to_csv(csv_path, index=False)
print(f"Wrote: {csv_path}  (rows={len(out)})")
print("Columns (snapshot):", [c for c in out.columns[:20]])
print("… total columns:", len(out.columns))

