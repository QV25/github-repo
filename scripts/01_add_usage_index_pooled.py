# -*- coding: utf-8 -*-
"""
01_add_usage_index_pooled.py
Adds pooled z-scores (Q7_mid, Q8_mid, Q9_mid) and usage_index to Sdon_clean.csv and Llogs_clean.csv.
"""

import os
import numpy as np
import pandas as pd

BASE = os.getcwd()
DERIVED = os.path.join(BASE, "results", "derived")
SDON = os.path.join(DERIVED, "Sdon_clean.csv")
LLOG = os.path.join(DERIVED, "Llogs_clean.csv")

def _zscore(x, mu, sd):
    x = pd.to_numeric(x, errors="coerce")
    if sd <= 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - mu) / sd

def main():
    if not (os.path.exists(SDON) and os.path.exists(LLOG)):
        raise FileNotFoundError("Missing Sdon_clean.csv or Llogs_clean.csv in results/derived/")

    sdon = pd.read_csv(SDON)
    llog = pd.read_csv(LLOG)

    for c in ["Q7_mid","Q8_mid","Q9_mid"]:
        sdon[c] = pd.to_numeric(sdon.get(c), errors="coerce")
        llog[c] = pd.to_numeric(llog.get(c), errors="coerce")

    pooled = pd.concat([sdon[["Q7_mid","Q8_mid","Q9_mid"]],
                        llog[["Q7_mid","Q8_mid","Q9_mid"]]], axis=0, ignore_index=True)

    mus = pooled.mean(skipna=True)
    sds = pooled.std(skipna=True, ddof=0)  # population sd for stability in small n

    for df in (sdon, llog):
        df["Q7_z"] = _zscore(df["Q7_mid"], mus["Q7_mid"], sds["Q7_mid"])
        df["Q8_z"] = _zscore(df["Q8_mid"], mus["Q8_mid"], sds["Q8_mid"])
        df["Q9_z"] = _zscore(df["Q9_mid"], mus["Q9_mid"], sds["Q9_mid"])
        df["usage_index"] = np.nanmean(df[["Q7_z","Q8_z","Q9_z"]].values, axis=1)

    sdon.to_csv(SDON, index=False)
    llog.to_csv(LLOG, index=False)

    print(f"[ok] Updated:\n  - {SDON}\n  - {LLOG}")
    print("Columns now include: Q7_z, Q8_z, Q9_z, usage_index")

if __name__ == "__main__":
    main()

