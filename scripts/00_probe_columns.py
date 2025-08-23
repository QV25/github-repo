import pandas as pd, json, pathlib, re
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "tables"
RESULTS.mkdir(parents=True, exist_ok=True)

def show_headers(path, tag):
    print(f"\n=== {tag}: {path.name} ===")
    raw2 = pd.read_csv(path, header=None, nrows=2, encoding="utf-8-sig", low_memory=False)
    lab_row = raw2.iloc[0].astype(str).tolist()
    fld_row = raw2.iloc[1].astype(str).tolist()
    print("header row 0 (labels):")
    for i, v in enumerate(lab_row):
        print(f"  [{i:02d}] {v!r}")
    print("header row 1 (fields):")
    for i, v in enumerate(fld_row):
        print(f"  [{i:02d}] {v!r}")
    # save mapping
    import csv
    out = RESULTS / f"columns_{tag}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index","field_name_row1","label_row0"])
        for i,(fld,lab) in enumerate(zip(fld_row, lab_row)):
            w.writerow([i, fld, lab])
    print(f"→ mapping saved to {out}")

    # try header=1 view
    try:
        df0 = pd.read_csv(path, header=1, nrows=0, encoding="utf-8-sig", low_memory=False)
        print(f"detected columns (header=1): {list(df0.columns)}")
    except Exception as e:
        print("header=1 failed:", e)

def guess_items_from_labels(mapping_csv, qkeys=("Q7","Q8","Q9","Q10","Q11","Q19","Q20")):
    import csv
    rows = list(csv.DictReader(open(mapping_csv, newline='', encoding="utf-8")))
    # patterns per item (NL+EN)
    pats = {
        "Q7":  r"(?i)(week|weekly|7\s*dagen|last\s*7)",
        "Q8":  r"(?i)(per\s*dag|/day|daily|op\s*een\s*dag|typical\s*day)",
        "Q9":  r"(?i)(duur|duration|minutes?|minuten|tijd\s*per\s*sessie)",
        "Q10": r"(?i)(dagen\s*actief|days\s*active|aantal\s*dagen)",
        "Q11": r"(?i)(context|use\s*case|type\s*gebruik|how\s*used)",
        "Q19": r"(?i)(belang|importance|important)",
        "Q20": r"(?i)(betalen|willing(ness)?\s*to\s*pay|pay|prijs|price)"
    }
    # search both field name and label text
    def find_for(key):
        rx = re.compile(pats[key])
        for r in rows:
            f = r["field_name_row1"] or ""
            l = r["label_row0"] or ""
            if rx.search(f) or rx.search(l):
                return r["field_name_row1"]
        # fallback: literal Q7.. in either
        for r in rows:
            f = r["field_name_row1"] or ""
            l = r["label_row0"] or ""
            if key in f or key in l:
                return r["field_name_row1"]
        return None

    print("\n=== GUESS: candidate columns per item (from labels/fields) ===")
    guess = {k: find_for(k) for k in qkeys}
    for k,v in guess.items():
        print(f"{k:>3}: {v!r}")
    return guess

# run
sf = DATA / "Survey full.csv"
sd = DATA / "Survey Data log full.csv"
dl = DATA / "Data logs.csv"

show_headers(sf, "survey_full")
show_headers(sd, "donor_survey")
show_headers(dl, "logs")

guess_items_from_labels(RESULTS / "columns_survey_full.csv")

