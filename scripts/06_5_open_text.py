import pathlib, json, re, unicodedata
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER, RES = ROOT/"derived", ROOT/"results"
TAB = RES/"tables"; TAB.mkdir(parents=True, exist_ok=True)

def load():
    S_all = pd.read_parquet(DER/"S_all.parquet")
    labels = json.load(open(DER/"labels_SurveyFull.json", encoding="utf-8"))
    return S_all, labels

# ---------- helpers ----------
OPEN_PATTERNS = [
    " - text", " other (text)", "please specify", "free text", "comment", "open",
    "if other", "explain", "why", " toelichting", " anders (tekst)", "overig (tekst)"
]

def detect_open_cols(S, labels):
    cands = []
    for col in S.columns:
        lab = str(labels.get(col, col)).lower()
        coln = str(col).lower()
        if any(p in lab for p in OPEN_PATTERNS) or any(p in coln for p in OPEN_PATTERNS):
            cands.append(col)
    # dedupe behoud volgorde
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def mask_pii(s: str) -> str:
    if not isinstance(s, str): return s
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", s)
    s = re.sub(r"\b(\+?\d[\d \-]{7,}\d)\b", "[phone]", s)
    return s

# bewaar korte domeintermen; stoplijst klein houden
STOP = set("""
a an the and or of in on for to from with without is are was were be being been
this that those these i you he she we they them our your his her it its
de het een en of in op voor tot van met zonder is zijn was waren ben bent
dit dat die deze ik jij hij zij wij jullie u hun ons jouw haar
""".split())

def tokens(text: str):
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[^\w\s\-]", " ", t)
    toks = [w for w in re.split(r"\s+", t) if w]
    # behoud 2+ tekens en filter alleen *algemene* stopwoorden
    toks = [w for w in toks if (len(w) >= 2 and w not in STOP)]
    return toks

def ngrams(ws, n):
    return [" ".join(ws[i:i+n]) for i in range(len(ws)-n+1)]

def best_example(theme: str, texts: list[str]) -> str:
    th_words = set(theme.split())
    best_txt, best_score = "", -1
    for txt in texts:
        tw = set(tokens(txt))
        sc = len(th_words & tw)
        if sc > best_score:
            best_score, best_txt = sc, txt
    return best_txt

def write_empty_stub(msg="No free-text themes could be derived."):
    pd.DataFrame(columns=["theme","n","share","example"]).to_csv(TAB/"open_themes.csv", index=False)
    (TAB/"H6_5_open_themes.tex").write_text(
        "\\begin{table}[t]\\centering\\small\\caption{Open responses}\\label{tab:6_5_open}"
        "\\begin{tabular}{lrrl}\\hline Theme & n & Share & Example \\\\ \\hline "
        + " \\multicolumn{4}{c}{"+msg.replace('_','\\_')+"} \\\\ \\hline\\end{tabular}\\end{table}\n",
        encoding="utf-8"
    )

def main():
    S, labels = load()
    open_cols = detect_open_cols(S, labels)
    if not open_cols:
        write_empty_stub("No free-text columns detected.")
        print("No open-text columns detected."); return

    texts = []
    for c in open_cols:
        texts += [mask_pii(str(x)).strip() for x in S[c].dropna().astype(str) if str(x).strip()]
    texts = [t for t in texts if t]
    if not texts:
        write_empty_stub("Open-text columns present, but all responses empty.")
        print("Open-text columns but empty."); return

    counts = {}
    for t in texts:
        ws = tokens(t)
        grams = ngrams(ws,1) + ngrams(ws,2)
        for g in grams:
            # negeer pure cijfers
            if re.fullmatch(r"\d+", g): continue
            counts[g] = counts.get(g, 0) + 1

    if not counts:
        write_empty_stub("No tokens after filtering; themes not applicable.")
        print("No tokens after filtering; wrote stub."); return

    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    N = len(texts)
    rows = []
    for theme, n in top:
        ex = best_example(theme, texts)
        ex = ex if len(ex) <= 120 else (ex[:120] + "…")
        rows.append({"theme": theme, "n": n, "share": n/N, "example": ex})
    OUT = pd.DataFrame(rows)
    OUT.to_csv(TAB/"open_themes.csv", index=False)

    # LaTeX
    def esc(s): 
        return s.replace("\\","\\textbackslash{}").replace("_","\\_").replace("%","\\%").replace("&","\\&").replace("#","\\#").replace("{","\\{").replace("}","\\}")
    show = OUT.copy()
    show["Share"] = (show["share"]*100).round(1)
    lines = ["\\begin{table}[t]","\\centering","\\small",
             "\\caption{Open responses: top n-grams as themes (with example quote)}",
             "\\label{tab:6_5_open}",
             "\\begin{tabular}{l r r l}\\hline",
             "Theme & n & Share (\\%) & Example \\\\ \\hline"]
    for _, r in show.iterrows():
        lines.append(f"{esc(r['theme'])} & {int(r['n'])} & {r['Share']:.1f} & {esc(r['example'])} \\\\")
    lines += ["\\hline","\\end{tabular}","\\end{table}"]
    (TAB/"H6_5_open_themes.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Written:")
    print("-", TAB/"open_themes.csv")
    print("-", TAB/"H6_5_open_themes.tex")

if __name__ == "__main__":
    main()

