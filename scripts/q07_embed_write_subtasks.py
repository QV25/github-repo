#!/usr/bin/env python3
"""
q07_embed_write_subtasks.py
-----------------------------------------------
Sub-tasks binnen *Writing & professional communication* (meerdere antwoorden)

Antwoorden:
 1  Outlining ideas or slides
 2  Drafting full text
 3  Proof-reading – tone adjustment
 4  Summarising sources or meeting notes
 5  Adjusting style for different audiences
 6  I did not choose “Writing & professional communication”

Werkt met embeddings (supersnel en goedkoop).
Output → results/answers_q07.xlsx
"""
import sys, pathlib, json, datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist

# ------------ project-modules ------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from embed_cache import EmbedCache
# -----------------------------------------------------------------------

PARSED   = pathlib.Path("parsed/all.jsonl")
PROTOS   = pathlib.Path("prototypes_q07.json")
OUT_XLS  = pathlib.Path("results/answers_q07.xlsx")
CACHE_NS = "q07_write_sub"

# 1. prototypes → embeddings -------------------------------------------
proto_texts = json.loads(PROTOS.read_text())
ec          = EmbedCache(CACHE_NS)
proto_vecs  = ec.get_embeddings(proto_texts)          # (6, D)

# 2. prompts verzamelen -------------------------------------------------
df  = pd.read_json(PARSED, lines=True)
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
dfu = dfu[dfu["prompt_short"] != ""]

# 3. prompt-embeddings ---------------------------------------------------
prompt_vecs = ec.get_embeddings(dfu["prompt_short"].tolist())   # (N, D)

# 4. cosine-similarity ---------------------------------------------------
sims   = 1 - cdist(prompt_vecs, proto_vecs, metric="cosine")    # (N, 6)
THRESH = 0.35
labels_per_prompt = []
for row in sims:
    top = row.max()
    lbls = [i + 1 for i, s in enumerate(row) if (s >= THRESH or s == top)]
    labels_per_prompt.append(lbls)
dfu["labels"] = labels_per_prompt

# 5. merge labels terug --------------------------------------------------
df = df.merge(dfu[["prompt_short", "labels"]],
              on="prompt_short", how="left")

# 6. explode + tellen ----------------------------------------------------
expl = (
    df.explode("labels")
      .dropna(subset=["labels"])
      .astype({"labels": "int"})
)
counts = (
    expl.groupby(["donor_id", "labels"])
        .size()
        .reset_index(name="n")
)
totals = expl.groupby("donor_id").size().rename("total")
counts = counts.merge(totals, on="donor_id")
counts = counts[(counts["n"] >= 3) & (counts["n"] / counts["total"] >= 0.05)]

# 7. donor-filter: alleen donors die *Writing* hadden -------------------
a6_path = pathlib.Path("results/answers_q06.xlsx")
if a6_path.exists():
    a6 = pd.read_excel(a6_path)
    writing_donors = set(
        a6[a6["category"].str.contains("Writing", na=False)]["donor_id"]
    )
else:
    writing_donors = set()
counts = counts[counts["donor_id"].isin(writing_donors)]

# 8. lijst per donor -----------------------------------------------------
valid = (
    counts.groupby("donor_id")["labels"]
          .apply(list)
          .reset_index()
)

mapping = {
    1: "Outlining ideas or slides",
    2: "Drafting full text",
    3: "Proof-reading – tone adjustment",
    4: "Summarising sources or meeting notes",
    5: "Adjusting style for different audiences",
    6: "I did not choose “Writing & professional communication”"
}

# → bouw eerst de |-gescheiden string uit labels
valid["category"] = valid["labels"].apply(
    lambda lst: "|".join(mapping[i] for i in sorted(lst))
)

# -- verwijder label 6 als donor óók andere subtaken heeft --------------
valid["category"] = (
    valid["category"]
      .str.replace(r"(^|[|])I did not choose .*?(?=$|[|])", "", regex=True)
      .str.strip("|")
)
valid = valid[valid["category"] != ""].copy()
valid.drop(columns="labels", inplace=True)
# ----------------------------------------------------------------------

# 9. donors zonder sub-labels → label 6 ---------------------------------
all_donors = set(df["donor_id"].unique())
with_subs  = set(valid["donor_id"])
no_subs    = all_donors - with_subs
if no_subs:
    extra = pd.DataFrame({
        "donor_id": list(no_subs),
        "category": mapping[6]
    })
    valid = pd.concat([valid, extra], ignore_index=True)

# 10. metadata + export --------------------------------------------------
valid = valid.sort_values("donor_id").reset_index(drop=True)
valid["survey_question"] = "q07_write_subtasks"
valid["timestamp"] = datetime.datetime.utcnow()

OUT_XLS.parent.mkdir(exist_ok=True)
valid.to_excel(OUT_XLS, index=False)
print(f"✅ klaar → {OUT_XLS}")

