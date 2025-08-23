#!/usr/bin/env python3
import json, argparse, pathlib, itertools

def _pairwise(msgs):
    it = iter(msgs)
    for q, a in itertools.zip_longest(it, it):
        if not a:
            break
        if q.get("author", {}).get("role") == "user" and \
           a.get("author", {}).get("role") == "assistant":
            yield q, a

def _load_messages(path):
    data = json.load(open(path, "r", encoding="utf8"))

    # 1) officiële export (dict met 'mapping')
    if isinstance(data, dict) and "mapping" in data:
        msgs = [n["message"] for n in data["mapping"].values()
                if n.get("message") and n["message"]["author"]["role"] in ("user","assistant")]
        msgs.sort(key=lambda m: m.get("create_time") or 0)
        return msgs

    # 2) export-per-conversatie (lijst)
    if isinstance(data, list):
        return [m for m in data if m.get("author", {}).get("role") in ("user","assistant")]

    raise ValueError(f"Onbekend JSON-formaat: {path}")

def parse_file(path: pathlib.Path):
    msgs = _load_messages(path)
    for idx, (q, a) in enumerate(_pairwise(msgs), 1):
        yield {
            "conversation_id": path.stem,
            "turn_index": idx,
            "question": " ".join(q.get("content", {}).get("parts", q.get("parts", []))),
            "answer":   " ".join(a.get("content", {}).get("parts", a.get("parts", []))),
            "question_time": q.get("create_time"),
            "answer_time":   a.get("create_time"),
        }

def main():
    ap = argparse.ArgumentParser(description="Parse ChatGPT export → JSONL")
    ap.add_argument("inputs", nargs="+", help="*.json bestanden (wildcards oké)")
    ap.add_argument("--out", required=True, help="uitvoer .jsonl")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf8") as fh:
        for pattern in args.inputs:
            for file in pathlib.Path().glob(pattern):
                for row in parse_file(file):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Klaar: {out} ({out.stat().st_size/1024:.1f} KB)")

if __name__ == "__main__":
    main()

