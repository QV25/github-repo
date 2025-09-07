#!/usr/bin/env python3
"""
embed_cache.py  –  helper om prompt-teksten goedkoop te embedden en lokaal te cachen

Gebruik:
    from embed_cache import EmbedCache
    ec = EmbedCache("q06_tasks")
    vecs = ec.get_embeddings(list_of_texts)   # np.ndarray (N x D)
"""

import os, json, hashlib, pathlib, time
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CACHE_DIR = pathlib.Path("intermediate")
CACHE_DIR.mkdir(exist_ok=True)

EMBED_MODEL             = "text-embedding-3-small"   # ~0.00002 $/token
BATCH_MAX_ITEMS         = int(os.getenv("EMB_MAX_ITEMS", "1000"))   # jij gebruikte 1000
BATCH_TOKENS_BUDGET     = int(os.getenv("EMB_TOK_BUDGET", "240000"))# ruim onder 300k cap
TOKENS_PER_CHAR         = float(os.getenv("EMB_TOK_PER_CHAR", "0.25"))  # ~4 chars/token
RATE_TOKENS_PER_SEC     = int(os.getenv("EMB_TOKENS_PER_SEC", "20000")) # throttle
MAX_BACKOFF_SECONDS     = 30

def sha1(txt: str) -> str:
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()

def _est_tokens(s: str) -> int:
    return max(1, int(len(s) * TOKENS_PER_CHAR))

class EmbedCache:
    def __init__(self, namespace: str):
        self.path = CACHE_DIR / f"{namespace}_embeddings.jsonl"
        self._load()

    # ---------- publiek --------------------------------------------------
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Retourneert NxD-matrix (np.float32). Vult automatisch de cache
        en retourneert in dezelfde volgorde als `texts`.
        """
        ids = [sha1(t) for t in texts]
        missing_items: List[Tuple[str, str]] = [(t, i) for t, i in zip(texts, ids) if i not in self.index]
        if missing_items:
            self._fetch_and_add(missing_items)

        vecs = np.vstack([self.index[i] for i in ids]).astype(np.float32)
        return vecs

    # ---------- intern ---------------------------------------------------
    def _load(self):
        self.index: dict[str, list[float]] = {}
        if self.path.exists():
            with self.path.open() as fh:
                for line in fh:
                    rec = json.loads(line)
                    self.index[rec["id"]] = rec["vec"]

    def _flush(self, new_records: list[dict]):
        with self.path.open("a") as fh:
            for rec in new_records:
                fh.write(json.dumps(rec) + "\n")

    def _fetch_and_add(self, missing_items: List[Tuple[str, str]]):
        """
        missing_items: list of (text, sha1_id)
        Batcht op tokenbudget en item-aantal. Throttlet requests (tokens/sec).
        Vangt 429/5xx/timeout/connection op met backoff.
        Bij BadRequest (400) wordt ALTIJD gesplitst (bisect) tot we het foute item isoleren.
        """
        new_recs: list[dict] = []
        pbar = tqdm(total=len(missing_items), desc="Embedding", unit="txt")

        def send_batch(batch: List[str], batch_ids: List[str], batch_num: int):
            if not batch:
                return
            # throttle o.b.v. geschat tokenvolume
            est_tokens = sum(_est_tokens(t) for t in batch)
            time.sleep(min(est_tokens / RATE_TOKENS_PER_SEC, 2.0))

            # probeer, en bij fouten handelen
            while True:
                try:
                    print(f"→ sending batch {batch_num}: {len(batch)} items (~{est_tokens} toks)")
                    res = openai.embeddings.create(model=EMBED_MODEL, input=batch)
                    for sid, emb in zip(batch_ids, res.data):
                        vec = emb.embedding
                        new_recs.append({"id": sid, "vec": vec})
                        self.index[sid] = vec
                    pbar.update(len(batch))
                    break
                except (openai.RateLimitError, openai.APIError, openai.APIConnectionError, openai.APITimeoutError) as e:
                    # exponentiële backoff
                    wait = getattr(send_batch, "_wait", 1.0)
                    print(f"   ↻ retry in {wait:.1f}s ({type(e).__name__})")
                    time.sleep(wait)
                    send_batch._wait = min(wait * 2, MAX_BACKOFF_SECONDS)
                except openai.BadRequestError as e:
                    # ALTIJD splitsen (bisect) om problematisch item te vinden
                    if len(batch) == 1:
                        # 1 item blijft falen → log en skip met minimale placeholder (veiligheid)
                        print(f"   ✖ BadRequest on single item (len={len(batch[0])} chars). Skipping this text.")
                        # sla NIETS op in cache voor dit id; caller kan later eventueel denoisen
                        pbar.update(1)
                        break
                    mid = len(batch) // 2
                    # stuur eerste helft
                    send_batch(batch[:mid], batch_ids[:mid], batch_num)
                    # stuur tweede helft
                    send_batch(batch[mid:], batch_ids[mid:], batch_num + 0.5)
                    break

        batch, batch_ids, batch_tokens = [], [], 0
        batch_num = 1

        for text, sid in missing_items:
            tks = _est_tokens(text)
            if batch and (len(batch) >= BATCH_MAX_ITEMS or batch_tokens + tks > BATCH_TOKENS_BUDGET):
                send_batch(batch, batch_ids, batch_num)
                batch, batch_ids, batch_tokens = [], [], 0
                batch_num += 1

            batch.append(text)
            batch_ids.append(sid)
            batch_tokens += tks

        if batch:
            send_batch(batch, batch_ids, batch_num)

        pbar.close()
        self._flush(new_recs)
