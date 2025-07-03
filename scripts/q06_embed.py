#!/usr/bin/env python3
"""
q06_embed.py  –  klassificeert prompts m.b.v. embeddings (supersnel + goedkoop)
Resultaat: results/answers_q06.xlsx (zelfde schema)
"""
import pathlib, json, datetime, sys
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from embed_cache import EmbedCache     # ligt in projectroot

PARSED   = pathlib.Path("parsed/all.jsonl")
PROTOS   = pathlib.Path("prototypes_q06.json")
OUT_XLS  = pathlib.Path("results/answers_q06.xlsx")
CACHE_NS = "q06_tasks"                 # eigen cache-bestand

# 1. prototypes inlezen + embedden
proto_texts = json.loads(PROTOS.read_text())
ec = EmbedCache(CACHE_NS)              # shared cache
proto_vecs  = ec.get_embeddings(proto_texts)     # (6, D)

# 2. unieke prompts inlezen
import pandas as pd
df = pd.read_json(PARSED, lines=True)
df["prompt_short"] = df["question"].str.slice(0, 400)
dfu = df.drop_duplicates("prompt_short").copy()

# -- prompt-cleaning -------------------------------------------------
dfu = df.drop_duplicates("prompt_short").copy()
dfu = dfu.dropna(subset=["prompt_short"]).copy()

# alles naar string, trim, en filter volledig lege regels
dfu["prompt_short"] = (
    dfu["prompt_short"]
      .astype(str)
      .str.replace(r"\s+", " ", regex=True)    # meerdere whitespace → spatie
      .str.strip()
)

dfu = dfu[dfu["prompt_short"] != ""].copy()    # verwijder lege strings
# -------------------------------------------------------------------


# 3. prompt-embeddings
prompt_vecs = ec.get_embeddings(dfu["prompt_short"].tolist())   # (N, D)

# 4. cosine similarity N×6
sims = 1 - cdist(prompt_vecs, proto_vecs, metric="cosine")      # (N, 6)

THRESH = 0.35
labels_per_prompt = []
for row in sims:
    top = row.max()
    lbls = [i+1 for i,sim in enumerate(row) if (sim >= THRESH or sim == top)]
    labels_per_prompt.append(lbls)

dfu["labels"] = labels_per_prompt

# 5. merge terug op hoofd-df
df = df.merge(dfu[["prompt_short","labels"]], on="prompt_short", how="left")

# 6. explode en tel zoals eerder
expl = (df.explode("labels")
          .dropna(subset=["labels"])
          .astype({"labels":"int"}))

counts = (expl.groupby(["donor_id","labels"])
                .size()
                .reset_index(name="n"))
totals = expl.groupby("donor_id").size().rename("total")
counts = counts.merge(totals, on="donor_id")
counts = counts[(counts["n"]>=5) & (counts["n"]/counts["total"]>=0.1)]

valid = (counts.groupby("donor_id")["labels"]
                .apply(list)
                .reset_index())

mapping = {
    1:"Writing & professional communication",
    2:"Brainstorming & personal ideas - fun",
    3:"Coding - programming help",
    4:"Language practice or translation",
    5:"Study revision - exam prep",
    6:"Other"
}
valid["category"] = valid["labels"].apply(
    lambda lst: "|".join(mapping[i] for i in sorted(lst))
)
valid["survey_question"] = "q06_tasks_used"
valid["timestamp"] = datetime.datetime.utcnow()
valid.drop(columns="labels", inplace=True)

OUT_XLS.parent.mkdir(exist_ok=True)
valid.to_excel(OUT_XLS, index=False)
print(f"✅ klaar → {OUT_XLS}")

