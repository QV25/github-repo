#!/usr/bin/env python3
"""
q08_embed_brain_subtasks.py
-------------------------------------------
Sub-tasks binnen *Brainstorming & personal ideas / fun*
(meerdere antwoorden mogelijk)

Antwoorden:
 1  Academic or research topics
 2  Business or marketing concepts
 3  Creative role-play, jokes, stories
 4  Hypothetical “what-if” scenarios
 5  Recommendations (books / movies / music)
 6  Trivia & general knowledge
 7  I did not choose “Brainstorming & personal ideas / fun”

Werkt met embeddings – snel & goedkoop.
Output → results/answers_q08.xlsx
"""
import sys, pathlib, json, datetime
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

# project-imports --------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from embed_cache import EmbedCache
# -----------------------------------------------------------------------

PARSED   = pathlib.Path("parsed/all.jsonl")
PROTOS   = pathlib.Path("prototypes_q08.json")
OUT_XLS  = pathlib.Path("results/answers_q08.xlsx")
CACHE_NS = "q08_brain_sub"

# 1. prototypes ----------------------------------------------------------
proto_texts = json.loads(PROTOS.read_text())
ec          = EmbedCache(CACHE_NS)
proto_vecs  = ec.get_embeddings(proto_texts)          # (7, D)

# 2. prompts -------------------------------------------------------------
df  = pd.read_json(PARSED, lines=True)
df["prompt_short"] = df["question"].str.slice(0, 400)

dfu = (
    df.drop_duplicates("prompt_short")
      .dropna(subset=["prompt_short"])
      .copy()
)
dfu["prompt_short"] = (
    dfu["prompt_short"].astype(str)
       .str.replace(r"\s+", " ", regex=True)
       .str.strip()
)
dfu = dfu[dfu["prompt_short"] != ""]

# 3. prompt-embeddings ---------------------------------------------------
prompt_vecs = ec.get_embeddings(dfu["prompt_short"].tolist())
sims   = 1 - cdist(prompt_vecs, proto_vecs, metric="cosine")    # (N, 7)

THRESH = 0.35
labels_per_prompt = []
for row in sims:
    top = row.max()
    labels = [i+1 for i, s in enumerate(row) if (s >= THRESH or s == top)]
    labels_per_prompt.append(labels)

dfu["labels"] = labels_per_prompt

# 4. merge labels terug --------------------------------------------------
df = df.merge(dfu[["prompt_short", "labels"]], on="prompt_short", how="left")

# 5. explode + tellen ----------------------------------------------------
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

# 6. donor-filter – alleen donors die Brainstorming hadden --------------
a6_path = pathlib.Path("results/answers_q06.xlsx")
if a6_path.exists():
    a6 = pd.read_excel(a6_path)
    brain_donors = set(
        a6[a6["category"].str.contains("Brainstorming", na=False)]["donor_id"]
    )
else:
    brain_donors = set()
counts = counts[counts["donor_id"].isin(brain_donors)]

# 7. lijst per donor -----------------------------------------------------
valid = (
    counts.groupby("donor_id")["labels"]
          .apply(list)
          .reset_index()
)

mapping = {
    1: "Academic or research topics",
    2: "Business or marketing concepts",
    3: "Creative role-play, jokes, stories",
    4: "Hypothetical what-if scenarios",
    5: "Recommendations (books, movies, music)",
    6: "Trivia & general knowledge",
    7: "I did not choose “Brainstorming & personal ideas / fun”"
}

valid["category"] = valid["labels"].apply(
    lambda lst: "|".join(mapping[i] for i in sorted(lst))
)

# exclusief label 7 ------------------------------------------------------
valid["category"] = (
    valid["category"]
      .str.replace(r"(^|[|])I did not choose .*?(?=$|[|])", "", regex=True)
      .str.strip("|")
)
valid = valid[valid["category"] != ""].copy()
valid.drop(columns="labels", inplace=True)

# 8. donors zonder sub-label → label 7 ----------------------------------
all_donors = set(df["donor_id"].unique())
with_subs  = set(valid["donor_id"])
no_subs    = all_donors - with_subs
if no_subs:
    extra = pd.DataFrame({
        "donor_id": list(no_subs),
        "category": mapping[7]
    })
    valid = pd.concat([valid, extra], ignore_index=True)

# 9. metadata + export ---------------------------------------------------
valid = valid.sort_values("donor_id").reset_index(drop=True)
valid["survey_question"] = "q08_brain_subtasks"
valid["timestamp"] = datetime.datetime.utcnow()

OUT_XLS.parent.mkdir(exist_ok=True)
valid.to_excel(OUT_XLS, index=False)
print(f"✅ klaar → {OUT_XLS}")

