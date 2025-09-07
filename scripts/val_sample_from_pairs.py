#!/usr/bin/env python3
import sys, re, json, random
from pathlib import Path
import pandas as pd

# ---------- CLI ----------
# Usage:
#   python3 scripts/val_sample_from_pairs.py \
#       --pairs results/q06_pairs.xlsx \
#       --jsonl parsed/all.jsonl \
#       --question Q12 \
#       --out validation/Q12_sample_100.csv \
#       --seed 2025 --min_donors 8 --per_donor 12 --extra 4
#
# Totals: min_donors*per_donor + extra = 100 (default 8*12 + 4)
# --------------------------------------

def parse_args(argv):
    args = {
        "pairs": None, "jsonl": "parsed/all.jsonl",
        "question": "Q12", "out": None,
        "seed": 2025, "min_donors": 8, "per_donor": 12, "extra": 4
    }
    i = 1
    while i < len(argv):
        if argv[i] == "--pairs": args["pairs"] = argv[i+1]; i+=2
        elif argv[i] == "--jsonl": args["jsonl"] = argv[i+1]; i+=2
        elif argv[i] == "--question": args["question"] = argv[i+1]; i+=2
        elif argv[i] == "--out": args["out"] = argv[i+1]; i+=2
        elif argv[i] == "--seed": args["seed"] = int(argv[i+1]); i+=2
        elif argv[i] == "--min_donors": args["min_donors"] = int(argv[i+1]); i+=2
        elif argv[i] == "--per_donor": args["per_donor"] = int(argv[i+1]); i+=2
        elif argv[i] == "--extra": args["extra"] = int(argv[i+1]); i+=2
        else:
            print(f"Unknown arg: {argv[i]}", file=sys.stderr); sys.exit(2)
    if not args["pairs"]:
        sys.exit("Missing --pairs results/<qXX>_pairs.xlsx")
    if not args["out"]:
        # default path under validation/
        p = Path(args["pairs"]).name.replace("_pairs.xlsx","").upper()
        args["out"] = f"validation/{p}_sample_100.csv"
    return args

# ---------- label code maps ----------
# Q12
MAP_Q12 = {
    "Writing & professional communication": "WRI",
    "Brainstorming & personal ideas - fun": "BRA",
    "Coding - programming help":            "COD",
    "Language practice or translation":     "LAN",
    "Study revision - exam prep":           "STU",
    "Other":                                "OTH",
    "No dominant task":                     "OTH"
}

# Q13 (writing subtasks)
MAP_Q13 = {
    "Creating an outline for a presentation or slide deck": "OUT",
    "Drafting a complete email, letter or report for me":   "DRAFT",
    "Proof-reading my text and correcting tone or grammar": "PROOF",
    "Summarising articles, sources or meeting notes":       "SUMM",
    "Rewriting the same text for different audiences":      "STYLE",
    "Any other use, or I did not choose Writing & professional communication": "OTH",
    "Other (writing-related)": "OTH"
}

# Q14 (brainstorm subtasks)
MAP_Q14 = {
    "Brainstorming academic or research ideas and paper topics":           "ACA",
    "Brainstorming business plans, product or marketing concepts":         "BUS",
    "Creative role-play, jokes or storytelling with ChatGPT":              "CRE",
    "Asking hypothetical what-if or alternate reality questions":          "HYP",
    "Requesting recommendations for books, movies or music":               "REC",
    "Asking trivia or general knowledge questions for fun":                "TRIV",
    "Any other use, or I did not choose Brainstorming & personal ideas":   "OTH",
    "Other (brainstorming-related)":                                       "OTH"
}

# Q15 (coding subtasks)
MAP_Q15 = {
    "Generate fresh code snippets or function templates for me": "GEN",
    "Debug my existing code and fix errors":                      "DBG",
    "Explain what a piece of code does or clarify a concept":    "EXP",
    "Convert code from one programming language to another":     "CONV",
    "Write sample unit tests for my functions":                  "TEST",
    "Any other use, or I did not choose Coding / programming help": "OTH",
    "Other (coding-related)": "OTH"
}

# Q16 (language subtasks)
MAP_Q16 = {
    "Translate an entire paragraph or document from one language into another": "TRANS",
    "Improve my grammar or writing style in the target language":               "GRAM",
    "Give me vocabulary drills or word lists to study":                         "VOCAB",
    "Do a conversational role-play so I can practise dialogue":                 "DIALOG",
    "Help with pronunciation or phonetic transcription":                        "PRON",
    "Any other use, or I did not choose Language practice or translation":      "OTH",
    "Other (language-related)": "OTH"
}

# Q17 (study subtasks)
MAP_Q17 = {
    "Summarise my lecture notes or textbook chapter concisely": "SUMM",
    "Generate practice questions or quizzes for my exam":       "QUIZ",
    "Explain a difficult concept to me in simple terms":        "EXPL",
    "Create mnemonics or other memory aids for key facts":      "MNEM",
    "Help me review flashcards or key terms for the test":      "FLASH",
    "Any other use, or I did not choose Study revision / exam prep": "OTH",
    "Other (study-related)": "OTH"
}

def pick_map(question):
    q = question.upper()
    if q == "Q12": return MAP_Q12
    if q == "Q13": return MAP_Q13
    if q == "Q14": return MAP_Q14
    if q == "Q15": return MAP_Q15
    if q == "Q16": return MAP_Q16
    if q == "Q17": return MAP_Q17
    raise ValueError(f"Unknown question: {question}")

def labels_to_codes(label_str, mapping):
    if not isinstance(label_str, str) or label_str.strip()=="":
        return ""
    parts = [p.strip() for p in label_str.split("|") if p.strip()!=""]
    codes = []
    for p in parts:
        code = mapping.get(p)
        if code is None:
            # probeer defensief te matchen zonder trailing 'or I did not choose...'
            p2 = re.sub(r"\s*or I did not choose.*$", "", p, flags=re.IGNORECASE).strip()
            code = mapping.get(p2, "OTH")
        codes.append(code)
    # unieke codes, volgorde stabiel
    seen, out = set(), []
    for c in codes:
        if c not in seen:
            out.append(c); seen.add(c)
    return ",".join(out)

def load_jsonl_answers(path):
    rows=[]
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append({
                "donor_id": str(obj.get("donor_id","")),
                "conversation_id": obj.get("conversation_id", None),
                "turn_index": obj.get("turn_index", None),
                "answer": str(obj.get("answer","")).strip(),
                "question": str(obj.get("question","")).strip()
            })
    return pd.DataFrame(rows)

def main():
    args = parse_args(sys.argv)
    random.seed(args["seed"])
    pairs = pd.read_excel(args["pairs"])
    if pairs.empty:
        sys.exit(f"[ERR] empty pairs file: {args['pairs']}")

    # merge answers
    base = load_jsonl_answers(args["jsonl"])
    pairs = pairs.merge(
        base[["donor_id","conversation_id","turn_index","answer"]],
        on=["donor_id","conversation_id","turn_index"],
        how="left"
    )
    mapping = pick_map(args["question"])

    # predicted → codes (comma‑sep)
    pairs["predicted_codes"] = pairs["labels"].apply(lambda s: labels_to_codes(s, mapping))

    # stratified sample: min_donors * per_donor + extra = 100
    donors_by_count = pairs["donor_id"].value_counts().index.tolist()
    picked_donors = donors_by_count[:args["min_donors"]] if len(donors_by_count)>=args["min_donors"] else donors_by_count

    chosen_idx = []
    for d in picked_donors:
        pool = pairs[pairs["donor_id"]==d]
        take = min(args["per_donor"], len(pool))
        chosen_idx += pool.sample(n=take, random_state=random.randint(1, 10_000)).index.tolist()

    # top‑up tot 100 – extra random
    need_total = args["min_donors"]*args["per_donor"] + args["extra"]
    if len(chosen_idx) < need_total:
        remaining = pairs.drop(index=chosen_idx)
        add = min(need_total - len(chosen_idx), len(remaining))
        chosen_idx += remaining.sample(n=add, random_state=args["seed"]).index.tolist()

    out = pairs.loc[chosen_idx].copy()
    # select & rename voor Excel‑template
    out = out.rename(columns={"labels":"predicted_labels"})
    out["gold_labels"] = ""  # leeg; jij vult dit in
    out = out[[
        "donor_id","conversation_id","turn_index",
        "question","answer","predicted_codes","predicted_labels","gold_labels"
    ]]
    out.insert(0, "sample_id", range(1, len(out)+1))

    Path(args["out"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args["out"], index=False, encoding="utf-8")
    print(f"[OK] wrote {args['out']}  (rows={len(out)})")

if __name__ == "__main__":
    main()

