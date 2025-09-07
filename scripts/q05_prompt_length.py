#!/usr/bin/env python3
"""
q05_prompt_length.py
--------------------
Q11: "How long are your typical prompts?"

Buckets:
  • One short sentence (≤ 20 words)
  • A short paragraph (21 – 60 words)
  • Multiple paragraphs (> 60 words)
  • Varies too much to say  (als geen bucket ≥ DOMINANCE)

Werkwijze:
  1) Woordcount per user-prompt (Unicode-woorden; codeblokken/backticks verwijderd)
  2) Bucket per prompt
  3) Verdeling per donor + shares
  4) Dominante bucket ≥ DOMINANCE → categorie; anders 'Varies too much to say'
"""

import pathlib, re, pandas as pd

PARSED  = pathlib.Path("parsed/all.jsonl")
OUT_XLS = pathlib.Path("results/answers_q05.xlsx")

DOMINANCE = 0.33  # *** 33% meerderheid ***

# Unicode-woord (incl. accenten/hyphens/apostrof)
WORD_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][0-9A-Za-zÀ-ÖØ-öø-ÿ]+)*")
CODE_BLOCK_RE = re.compile(r"```.*?```", flags=re.S)  # fenced code verwijderen
BACKTICK_RE   = re.compile(r"`+")                     # losse backticks verwijderen

def clean_for_wc(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = CODE_BLOCK_RE.sub(" ", text)
    text = BACKTICK_RE.sub(" ", text)
    return text

def word_count(text: str) -> int:
    return len(WORD_RE.findall(clean_for_wc(text)))

def to_bucket(n_words: int) -> str:
    if n_words <= 20:
        return "One short sentence (≤ 20 words)"
    elif n_words <= 60:
        return "A short paragraph (21 – 60 words)"
    else:
        return "Multiple paragraphs (> 60 words)"

# 1) Lees prompts en bepaal woordcount + bucket
df = pd.read_json(PARSED, lines=True)
df["word_count"] = df["question"].apply(word_count)
df["bucket"] = df["word_count"].apply(to_bucket)

# 2) Verdeling per donor (counts) en shares
counts = df.pivot_table(index="donor_id", columns="bucket",
                        values="turn_index", aggfunc="count", fill_value=0)

for col in ["One short sentence (≤ 20 words)",
            "A short paragraph (21 – 60 words)",
            "Multiple paragraphs (> 60 words)"]:
    if col not in counts.columns:
        counts[col] = 0

counts = counts.reset_index()
counts.rename(columns={
    "One short sentence (≤ 20 words)": "cnt_sentence_le20",
    "A short paragraph (21 – 60 words)": "cnt_paragraph_21_60",
    "Multiple paragraphs (> 60 words)": "cnt_multi_gt60",
}, inplace=True)

counts["n_prompts_total"] = (
    counts["cnt_sentence_le20"] + counts["cnt_paragraph_21_60"] + counts["cnt_multi_gt60"]
)

den = counts["n_prompts_total"].where(counts["n_prompts_total"]>0, other=1)
counts["share_sentence"]  = counts["cnt_sentence_le20"]   / den
counts["share_paragraph"] = counts["cnt_paragraph_21_60"] / den
counts["share_multi"]     = counts["cnt_multi_gt60"]      / den

def decide(row):
    if row["n_prompts_total"] == 0:
        return "Varies too much to say"
    shares = {
        "One short sentence (≤ 20 words)": row["share_sentence"],
        "A short paragraph (21 – 60 words)": row["share_paragraph"],
        "Multiple paragraphs (> 60 words)": row["share_multi"],
    }
    top_label = max(shares, key=shares.get)
    if shares[top_label] >= DOMINANCE:
        return top_label
    return "Varies too much to say"

counts["category"] = counts.apply(decide, axis=1)
counts["survey_question"]     = "q05_prompt_length"
counts["dominance_threshold"] = DOMINANCE

out = counts[[ "donor_id", "n_prompts_total",
               "cnt_sentence_le20", "cnt_paragraph_21_60", "cnt_multi_gt60",
               "share_sentence", "share_paragraph", "share_multi",
               "category", "survey_question", "dominance_threshold" ]]

OUT_XLS.parent.mkdir(exist_ok=True, parents=True)
mode = "a" if OUT_XLS.exists() else "w"
writer_kwargs = {"mode": mode, "engine": "openpyxl"}
if mode == "a":
    writer_kwargs["if_sheet_exists"] = "overlay"
with pd.ExcelWriter(OUT_XLS, **writer_kwargs) as xls:
    out.to_excel(xls, index=False, header=(mode == "w"))

print(f"✅ Q11 geschreven → {OUT_XLS} (n_donors={len(out)})")
