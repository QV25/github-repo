#!/usr/bin/env python3
"""
embed_cache.py  –  helper om prompt-teksten goedkoop te embedden en lokaal te cachen
Usage (vanuit andere scripts):
    from embed_cache import EmbedCache
    ec = EmbedCache("q06_tasks")        # eigen naam → eigen cache-file
    vecs = ec.get_embeddings(list_of_texts)   # np.ndarray
"""

import os, json, hashlib, pathlib, math, time
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CACHE_DIR = pathlib.Path("intermediate")
CACHE_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"            #  ~0.00002 $/token
BATCH_SIZE  = 1000                                # max 2048 voor dit model

def sha1(txt: str) -> str:
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()

class EmbedCache:
    def __init__(self, namespace: str):
        self.path = CACHE_DIR / f"{namespace}_embeddings.jsonl"
        self._load()

    # ---------- publiek --------------------------------------------------
    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Retourneert NxD-matrix (np.float32). Vult automatisch de cache
        en retourneert in dezelfde volgorde als `texts`.
        """
        missing = [t for t in texts if sha1(t) not in self.index]
        if missing:
            self._fetch_and_add(missing)

        # build matrix in volgorde
        vecs = np.vstack([self.index[sha1(t)] for t in texts]).astype(np.float32)
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
        """Append nieuwe records naar disk."""
        with self.path.open("a") as fh:
            for rec in new_records:
                fh.write(json.dumps(rec) + "\n")

    def _fetch_and_add(self, texts: list[str]):
        new_recs = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
            batch = texts[i : i + BATCH_SIZE]
            res = openai.embeddings.create(
                model=EMBED_MODEL,
                input=batch,
            )
            # API retourneert in dezelfde volgorde
            for txt, emb in zip(batch, res.data):
                new_recs.append({"id": sha1(txt), "vec": emb.embedding})
                self.index[sha1(txt)] = emb.embedding
            time.sleep(0.5)           # mini-pauze; voorkomt rate-limit spikes
        self._flush(new_recs)

