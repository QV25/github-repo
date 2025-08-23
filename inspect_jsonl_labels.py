#!/usr/bin/env python3
import json, re
from pathlib import Path
from collections import Counter, defaultdict

PATH = Path("parsed/all.jsonl")
N = 2000   # lees tot 2000 regels voor een snel beeld

def norm(s): return re.sub(r"\s+", " ", str(s)).strip()

def main():
    if not PATH.exists():
        print(f"[ERR] not found: {PATH}"); return
    keys_seen = Counter()
    samples = []
    with PATH.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if i > N: break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            samples.append(obj)
            for k in obj.keys(): keys_seen[k]+=1

    print("== top keys ==")
    for k,c in keys_seen.most_common(30): print(f"{k:25s} {c}")

    # vind kansrijke labelvelden per vraag
    for q in ["q12","q13"]:
        cand = [k for k in keys_seen if re.search(fr"^{q}(_|$)", k, flags=re.I) or re.search(fr"{q}_labels", k, flags=re.I)]
        print(f"\n== candidates for {q} ==")
        print(", ".join(cand) if cand else "(none)")

        # toon top labelstrings indien list/str
        counts = Counter()
        for obj in samples:
            for k in cand:
                v = obj.get(k)
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, str) and it.strip(): counts[norm(it)] += 1
                elif isinstance(v, str) and v.strip():
                    # split op ; of , voor multi-select strings
                    parts = [t.strip() for t in re.split(r"[;,]", v) if t.strip()]
                    for it in parts: counts[norm(it)] += 1
                elif isinstance(v, dict):
                    # booleans als one-hot?
                    for kk,val in v.items():
                        if val in (True, 1, "true", "True"): counts[norm(kk)] += 1
        if counts:
            print(f"== top labels for {q} (first 20) ==")
            for lab, c in counts.most_common(20): print(f"{c:5d}  {lab}")
        else:
            print(f"(no obvious labels extracted for {q})")

if __name__ == "__main__":
    main()

