#!/usr/bin/env python3
"""
chatgpt_parser.py — ChatGPT conversations.json → JSONL (1 Q+A per regel)

- Invoer: bestanden, globs of mappen. Voor mappen recursief zoeken naar:
  • conversations*.json
  • conversation.json   (let op: NIET conversation*.json, om doublures te voorkomen)
- Eén Q+A = consecutive user → assistant
- Q+A zonder bruikbare timestamps (vraag of antwoord) worden SKIPPED
- Optioneel: PII-redactie (--redact) met tellingsrapport (--pii-log CSV)
- Tijden blijven UNIX seconds (UTC); conversie doen we downstream
"""

import json, argparse, pathlib, itertools, hashlib, glob, re, csv
from pathlib import Path
from collections import defaultdict

# --------------------------- helpers --------------------------------

def _text_from_part(part):
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        return part.get('text') or ""
    return str(part)

def anon_id(path: Path) -> str:
    # 8-char hash uit bestandsnaam: stabiel binnen jouw project
    return hashlib.sha256(path.stem.encode()).hexdigest()[:8]

def msg_text(msg):
    content = msg.get('content', {})
    if isinstance(content, dict):
        if 'parts' in content:
            return " ".join(_text_from_part(p) for p in content['parts']).strip()
        if 'text' in content:
            return str(content['text']).strip()
    if isinstance(content, list):
        return " ".join(_text_from_part(p) for p in content).strip()
    return str(content).strip()

def safe_ts(msg):
    t = msg.get('create_time') or msg.get('update_time')
    try:
        return float(t)
    except Exception:
        return None

def is_ua(m):
    return m and m.get('author', {}).get('role') in ('user', 'assistant')

def pairwise_user_assistant(msgs):
    """yield (user, assistant) paren uit chronologisch gesorteerde berichten"""
    it = iter(msgs)
    for q, a in itertools.zip_longest(it, it):
        if not a:
            break
        if is_ua(q) and is_ua(a) and q['author']['role'] == 'user' and a['author']['role'] == 'assistant':
            yield q, a

# ------------------ verschillende exportstructuren -------------------

def _from_mapping(mapping):
    msgs = [n['message'] for n in mapping.values() if is_ua(n.get('message'))]
    # sorteer op tijd; ontbrekend -> 0.0 zodat sort geen error geeft
    msgs.sort(key=lambda m: safe_ts(m) or 0.0)
    return msgs

def load_conversations(path: Path):
    with open(path, 'r', encoding='utf8') as fh:
        data = json.load(fh)

    # 1) dict met 'mapping'
    if isinstance(data, dict) and 'mapping' in data:
        yield path.stem, _from_mapping(data['mapping'])
        return

    # 2) lijst van messages
    if isinstance(data, list) and data and is_ua(data[0]):
        msgs = [m for m in data if is_ua(m)]
        msgs.sort(key=lambda m: safe_ts(m) or 0.0)
        yield path.stem, msgs
        return

    # 3) lijst van conversaties (elk met mapping)
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'mapping' in data[0]:
        for i, conv in enumerate(data):
            cid = conv.get('id') or f"{path.stem}_{i}"
            yield cid, _from_mapping(conv['mapping'])
        return

    raise ValueError(f"Onbekend JSON-formaat: {path}")

# ------------------------ PII redactie -------------------------------

# Regex-patronen (bewust conservatief; geen persoonsnamen via regex)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
RE_URL   = re.compile(r"\b(?:(?:https?://|www\.)\S+)\b", re.IGNORECASE)
RE_IBAN  = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){10,30}\b")
RE_CARD  = re.compile(r"\b(?:\d[ -]?){13,19}\b")
RE_PHONE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{2,4}\)?[\s\-\.]?)?\d{3,4}[\s\-\.]?\d{3,4}(?!\d)"
)
RE_IP    = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.){3}(?:25[0-5]|2[0-4]\d|1?\d{1,2})\b"
)

PII_PATTERNS = [
    ("EMAIL", RE_EMAIL, "[EMAIL]"),
    ("URL",   RE_URL,   "[URL]"),
    ("IBAN",  RE_IBAN,  "[IBAN]"),
    ("CARD",  RE_CARD,  "[CARD]"),
    ("PHONE", RE_PHONE, "[PHONE]"),
    ("IP",    RE_IP,    "[IP]"),
]

def redact_text(text: str, counters: dict) -> str:
    """Vervang PII-patronen door tokens en tel vervangingen per type."""
    if not text:
        return text
    s = text
    for key, rx, token in PII_PATTERNS:
        s, n = rx.subn(token, s)
        if n:
            counters[key] += n
    return s

# ----------------------- input-bestanden vinden ----------------------

def iter_input_files(inputs):
    """accepteert bestanden, globs en mappen; levert unieke Path-objecten op"""
    seen = set()
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            # recursief in de map — let op: geen 'conversation*.json' om doublures te voorkomen
            for pat in ("conversations*.json", "conversation.json"):
                for fp in p.rglob(pat):
                    seen.add(fp.resolve())
        else:
            # glob/bestand
            for match in glob.glob(str(inp), recursive=True):
                mp = Path(match)
                if mp.is_file():
                    seen.add(mp.resolve())
    return sorted(seen)

# ----------------------------- CLI ----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Parse ChatGPT export → JSONL (1 Q+A per regel)")
    ap.add_argument("inputs", nargs="+", help="paden: bestanden, globs of mappen (bv. data/raw)")
    ap.add_argument("--out", required=True, help="uitvoer .jsonl")
    ap.add_argument("--redact", action="store_true", help="activeer PII-redactie op question/answer")
    ap.add_argument("--pii-log", default=None, help="pad naar CSV met PII-tellingen per donor/type")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(args.inputs)
    if not files:
        print("⚠️  Geen invoerbestanden gevonden. Controleer pad/patroon.")
        return

    total_rows = 0
    total_drops = 0

    # aggregatie voor PII-tellingen
    pii_totals = defaultdict(int)                 # per type
    pii_by_donor = defaultdict(lambda: defaultdict(int))  # donor_id -> type -> count

    with out.open('w', encoding='utf8') as fh:
        for file in files:
            try:
                for conv_id, msgs in load_conversations(file):
                    idx = 0
                    for q, a in pairwise_user_assistant(msgs):
                        qt, at = safe_ts(q), safe_ts(a)
                        if qt is None or at is None:
                            total_drops += 1
                            continue

                        qtxt = msg_text(q)
                        atxt = msg_text(a)

                        donor = anon_id(file)

                        if args.redact:
                            local_counts = defaultdict(int)
                            qtxt = redact_text(qtxt, local_counts)
                            atxt = redact_text(atxt, local_counts)
                            if local_counts:
                                for k, v in local_counts.items():
                                    pii_totals[k] += v
                                    pii_by_donor[donor][k] += v

                        idx += 1
                        row = {
                            "donor_id": donor,
                            "conversation_id": conv_id,
                            "turn_index": idx,
                            "question": qtxt,
                            "answer": atxt,
                            "question_time": qt,   # UNIX (UTC)
                            "answer_time": at,     # UNIX (UTC)
                        }
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        total_rows += 1
            except Exception as e:
                print(f"⚠️  Overslaan vanwege parse-fout: {file} ({e})")

    # optioneel: pii-log naar CSV
    if args.redact and args.pii_log:
        pii_path = Path(args.pii_log)
        pii_path.parent.mkdir(parents=True, exist_ok=True)
        with pii_path.open("w", newline="", encoding="utf8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["donor_id", "pii_type", "replacements"])
            for donor, d in sorted(pii_by_donor.items()):
                for typ, cnt in sorted(d.items()):
                    writer.writerow([donor, typ, cnt])

    sz_kb = out.stat().st_size/1024 if out.exists() else 0
    print(f"📄 Bestanden gevonden: {len(files)}")
    print(f"✅ {total_rows} Q+A-paren → {out} ({sz_kb:.1f} KB)")
    print(f"ℹ️  Q+A gedropt (ontbrekende timestamps): {total_drops}")
    if args.redact:
        print("🔒 PII-redactie actief — vervangingen (totaal):")
        for typ in ["EMAIL","URL","IBAN","CARD","PHONE","IP"]:
            if pii_totals.get(typ, 0):
                print(f"   {typ}: {pii_totals[typ]}")
        if args.pii_log:
            print(f"📝 PII-log geschreven naar: {args.pii_log}")

if __name__ == "__main__":
    main()
