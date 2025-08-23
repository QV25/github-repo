#!/usr/bin/env python3
"""
audit_ch4_consistency.py — Controleer of outputs q06–q11 logisch zijn t.o.v. hoofdstuk 4

Doet o.a.:
- Vergelijkt donor-sets q06 vs subtaken (q07–q11).
- Checkt of subtaken alleen voorkomen bij donors die de hoofdcategorie kozen (q06).
- Telt 'I did not choose …' combinaties (zou nooit met andere labels samen mogen).
- Samenvattingen per bestand.
"""

import pathlib, re
import pandas as pd

RES = pathlib.Path("results")

PARENTS = {
    7:  "Writing & professional communication",
    8:  "Brainstorming & personal ideas - fun",
    9:  "Coding - programming help",
    10: "Language practice or translation",
    11: "Study revision - exam prep",
}

def canon(s: str) -> str:
    s = str(s or "").strip().lower()
    s = s.replace("—","-").replace("–","-").replace("“",'"').replace("”",'"')
    s = re.sub(r"\s*[/\-]\s*", " - ", s)   # unify "/" and "-" as " - "
    s = re.sub(r"\s+", " ", s)
    return s

def donors_with_parent(df6: pd.DataFrame, parent: str) -> set[str]:
    target = canon(parent)
    out = set()
    for _, row in df6.iterrows():
        parts = [canon(x) for x in str(row["category"]).split("|")]
        if target in parts:
            out.add(str(row["donor_id"]))
    return out

def load_answers(q: int) -> pd.DataFrame:
    f = RES / f"answers_q{q:02d}.xlsx"
    df = pd.read_excel(f)
    df["donor_id"] = df["donor_id"].astype(str)
    df["category"] = df["category"].astype(str)
    return df

def main():
    # basis
    df6 = load_answers(6)
    donors_all_q6 = set(df6["donor_id"])

    # sanity q06
    print(f"q06 donors: {len(donors_all_q6)}")
    other_mix = df6[df6["category"].str.contains("Other (no dominant task)") & df6["category"].str.contains("\\|")]
    if not other_mix.empty:
        print("⚠️ In q06 staat 'Other (no dominant task)' samen met andere labels (mag niet):")
        print(other_mix[["donor_id","category"]])

    # per subtaak-bestand
    for q in (7,8,9,10,11):
        df = load_answers(q)
        parent = PARENTS[q]
        parent_donors = donors_with_parent(df6, parent)

        donors_sub = set(df["donor_id"])
        outside = sorted(donors_sub - parent_donors)
        missing = sorted(parent_donors - donors_sub)

        print(f"\n=== q{q:02d} vs parent '{parent}' ===")
        print(f"  donors in subtasks: {len(donors_sub)}")
        print(f"  donors met parent in q06: {len(parent_donors)}")
        if outside:
            print(f"  ⚠️ {len(outside)} donors in q{q:02d} die géén '{parent}' hebben in q06:", outside[:10])
        else:
            print("  ✅ geen donors buiten parent-set")

        # 'I did not choose …' mag niet gecombineerd zijn met andere subtaken
        bad_none_combo = df[df["category"].str.contains("I did not choose") & df["category"].str.contains("\\|")]
        if not bad_none_combo.empty:
            print("  ⚠️ 'I did not choose …' gecombineerd met andere labels (mag niet):")
            print(bad_none_combo.head())

        # kleine samenvatting per optie
        counts = df["category"].value_counts().sort_values(ascending=False)
        print("\n  Top categories / rows:")
        print(counts.head(10))

    print("\n✅ Audit klaar.")
    
if __name__ == "__main__":
    main()

