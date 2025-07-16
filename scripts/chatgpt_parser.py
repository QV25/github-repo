#!/usr/bin/env python3
"""chatgpt_parser.py – simple but robust ChatGPT export → JSONL

Handles 3 common export shapes and tolerates `parts` that are lists of
strings **or** dicts (e.g. code blocks). Suitable for < 1 GB files in
memory. Writes one JSON object per Q‑A pair.
"""

import json, argparse, pathlib, itertools, sys, hashlib

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_from_part(part):
    """Return plain text regardless of part structure."""
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        # official export code blocks are dicts with 'text'
        return part.get('text') or ""
    return str(part)


def anon_id(path: pathlib.Path) -> str:
    """8-karakter hash uit bestandsnaam → stabiele, anonieme ID."""
    return hashlib.sha256(path.stem.encode()).hexdigest()[:8]


def msg_text(msg):
    """Concatenate parts or legacy `text` field into one string."""
    content = msg.get('content', {})
    if isinstance(content, dict):
        if 'parts' in content:
            parts = content['parts']
            return " ".join(_text_from_part(p) for p in parts).strip()
        if 'text' in content:
            return str(content['text']).strip()
    if isinstance(content, list):
        return " ".join(_text_from_part(p) for p in content).strip()
    return str(content).strip()


def msg_time(m):
    return m.get('create_time') or m.get('update_time') or 0


def is_ua(m):
    return m and m.get('author', {}).get('role') in ('user', 'assistant')


def pairwise(msgs):
    it = iter(msgs)
    for q, a in itertools.zip_longest(it, it):
        if not a:
            break
        if is_ua(q) and is_ua(a) and q['author']['role'] == 'user':
            yield q, a

# ---------------------------------------------------------------------------
# Extract message-lists from different export structures
# ---------------------------------------------------------------------------

def _from_mapping(mapping):
    msgs = [n['message'] for n in mapping.values() if is_ua(n.get('message'))]
    return sorted(msgs, key=msg_time)


def load_conversations(path):
    data = json.load(open(path, 'r', encoding='utf8'))

    # 1) dict with 'mapping' key (official full export)
    if isinstance(data, dict) and 'mapping' in data:
        yield path.stem, _from_mapping(data['mapping'])
        return

    # 2) list of messages (per‑conversation export)
    if isinstance(data, list) and data and is_ua(data[0]):
        yield path.stem, data
        return

    # 3) list of conversations (each item has a mapping)
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'mapping' in data[0]:
        for i, conv in enumerate(data):
            cid = conv.get('id') or f"{path.stem}_{i}"
            yield cid, _from_mapping(conv['mapping'])
        return

    raise ValueError(f"Onbekend JSON-formaat: {path}")

# ---------------------------------------------------------------------------
# Generator voor Q‑A regels
# ---------------------------------------------------------------------------

def parse_file(path: pathlib.Path):
    for conv_id, msgs in load_conversations(path):
        for idx, (q, a) in enumerate(pairwise(msgs), 1):
            yield {
                "donor_id": anon_id(path),
		"conversation_id": conv_id,
                "turn_index": idx,
                "question": msg_text(q),
                "answer": msg_text(a),
                "question_time": msg_time(q),
                "answer_time": msg_time(a),
            }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Parse ChatGPT export → JSONL (1 Q-A per regel)")
    ap.add_argument("inputs", nargs="+", help=".json bestanden (wildcards oké)")
    ap.add_argument("--out", required=True, help="uitvoer .jsonl")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out.open('w', encoding='utf8') as fh:
        for pattern in args.inputs:
            for file in pathlib.Path().glob(pattern):
                for row in parse_file(file):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1

    print(f"✅ {n} Q-A-paren → {out} ({out.stat().st_size/1024:.1f} KB)")

if __name__ == "__main__":
    main()






































































