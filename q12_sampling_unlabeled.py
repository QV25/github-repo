#!/usr/bin/env python3
import json, random, sys
from pathlib import Path
import pandas as pd

# ---- CONFIG ----
ALL_JSONL = Path("parsed/all.jsonl")         # bronbestand
OUT_CSV   = Path("q12_sample_100_unlabeled.csv")
RANDOM_SEED = 2025

DONOR_FIELD    = "donor_id"
QUESTION_FIELD = "question"
ANSWER_FIELD   = "answer"

TARGET_DONORS  = 6      # aantal verschillende donors in de "blok"
PER_DONOR      = 16     # rijen per donor (6*16 = 96)
EXTRA_RANDOM   = 4      # extra random rijen (totaal = 100)
# ----------------

random.seed(RANDOM_SEED)

def load_jsonl(p: Path) -> pd.DataFrame:
    if not p.exists():
        sys.exit(f"[ERR] not found: {p}")
    rows = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append({
                "donor_id":      str(obj.get(DONOR_FIELD, "")),
                "question_text": str(obj.get(QUESTION_FIELD, "")).strip(),
                "answer_text":   str(obj.get(ANSWER_FIELD, "")).strip()
            })
    df = pd.DataFrame(rows)
    # filter niet-lege Q/A en drop exacte duplicaten
    df = df[(df["question_text"] != "") & (df["answer_text"] != "")]
    df = df.drop_duplicates(subset=["donor_id","question_text","answer_text"])
    if df.empty:
        sys.exit("[ERR] geen bruikbare rijen in parsed/all.jsonl")
    return df

def main():
    df = load_jsonl(ALL_JSONL)

    # kies donors (top op basis van aantal rijen)
    donor_counts = df["donor_id"].value_counts()
    donors = donor_counts.index.tolist()
    picked = donors[:TARGET_DONORS] if len(donors) >= TARGET_DONORS else donors
    if len(picked) < TARGET_DONORS:
        print(f"[WARN] slechts {len(picked)} donors gevonden; vullen aan met extra random rijen.", file=sys.stderr)

    # verzamel per-donor (max PER_DONOR)
    chosen_idx = []
    for d in picked:
        pool = df[df["donor_id"] == d]
        take_n = min(PER_DONOR, len(pool))
        chosen_idx += pool.sample(n=take_n, random_state=random.randint(1, 10_000)).index.tolist()

    # top up tot 96 als een donor < PER_DONOR had
    target_block_total = TARGET_DONORS * PER_DONOR
    if len(chosen_idx) < target_block_total:
        remaining = df.drop(index=chosen_idx)
        add_needed = min(target_block_total - len(chosen_idx), len(remaining))
        chosen_idx += remaining.sample(n=add_needed, random_state=random.randint(1, 10_000)).index.tolist()

    # 4 extra random die nog niet gekozen zijn
    remaining = df.drop(index=chosen_idx)
    extra_idx = remaining.sample(n=min(EXTRA_RANDOM, len(remaining)),
                                 random_state=random.randint(1, 10_000)).index.tolist()

    # assembleer en borg 100 rijen
    block = df.loc[chosen_idx].copy(); block["source"] = "donor-block"
    extra = df.loc[extra_idx].copy();  extra["source"] = "random"
    out = pd.concat([block, extra], ignore_index=True)

    # als >100: trim; als <100: aanvullen
    if len(out) > 100:
        out = out.sample(n=100, random_state=RANDOM_SEED)
    elif len(out) < 100:
        rem = df.drop(index=out.index, errors="ignore")
        need = min(100 - len(out), len(rem))
        out = pd.concat([out, rem.sample(n=need, random_state=random.randint(1, 10_000))],
                        ignore_index=True)

    # shuffle stabiel, index en export
    out = out.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    out.insert(0, "item_id", out.index + 1)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"Written: {OUT_CSV.resolve()}    Rows: {len(out)}")
    print(f"Donors used (up to {TARGET_DONORS}): {picked[:TARGET_DONORS]}")

if __name__ == "__main__":
    main()

