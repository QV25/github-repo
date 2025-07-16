#!/usr/bin/env python3
"""
q06_tasks_used.py
-----------------
Beantwoordt survey-vraag 6 (meerdere antwoorden) met GPT.

Output → results/answers_q06.xlsx
"""
import sys, pathlib, datetime, json
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from llm_utils import gpt

PARSED  = pathlib.Path("parsed/all.jsonl")
OUT_XLS = pathlib.Path("results/answers_q06.xlsx")

PROMPT = (
    "You will be given a user's prompt to ChatGPT.\n"
    "Assign **ALL** applicable categories from this fixed list:\n"
    "1. Writing & professional communication\n"
    "2. Brainstorming & personal ideas - fun\n"
    "3. Coding - programming help\n"
    "4. Language practice or translation\n"
    "5. Study revision - exam prep\n"
    "6. Other\n\n"
    "Respond ONLY with a JSON array of integers, e.g. [1,3].\n"
    "Prompt: \"\"\"\n{user_text}\n\"\"\""
)

# 1. Lees Q-prompts
df = pd.read_json(PARSED, lines=True)
df["prompt_short"] = df["question"].str.slice(0, 400)

# 2. GPT-label per prompt (cached)  ← **één keer**
def label(text):
    resp = gpt(PROMPT.format(user_text=text), temperature=0, max_tokens=20)
    try:
        return json.loads(resp)
    except json.JSONDecodeError:
        return []

df["labels"] = [label(t) for t in tqdm(df["prompt_short"], desc="GPT classify")]


# 3. Explode → één label per regel
exploded = (
    df.explode("labels")
      .dropna(subset=["labels"])
      .astype({"labels": "int"})
)

# 4. Tellen per donor per label
counts = (
    exploded.groupby(["donor_id", "labels"])
            .size()
            .reset_index(name="n")
)

# 5. Filter op drempels
totals = exploded.groupby("donor_id").size().rename("total")
counts = counts.merge(totals, on="donor_id")
counts = counts[(counts["n"] >= 3) & (counts["n"] / counts["total"] >= 0.05)]

# 6. Verzamel geldige labels per donor
valid = (
    counts.groupby("donor_id")["labels"]
          .apply(list)
          .reset_index()
)

def map_labels(lst):
    mapping = {
        1: "Writing & professional communication",
        2: "Brainstorming & personal ideas - fun",
        3: "Coding - programming help",
        4: "Language practice or translation",
        5: "Study revision - exam prep",
        6: "Other",
    }
    return "|".join(mapping[i] for i in sorted(lst))

valid["category"]        = valid["labels"].apply(map_labels)
valid["survey_question"] = "q06_tasks_used"
valid["timestamp"]       = datetime.datetime.utcnow()
valid.drop(columns="labels", inplace=True)

print(valid.head())

# 7. Wegschrijven
OUT_XLS.parent.mkdir(exist_ok=True)
valid.to_excel(OUT_XLS, index=False)
print(f"✅ Resultaten → {OUT_XLS}")

