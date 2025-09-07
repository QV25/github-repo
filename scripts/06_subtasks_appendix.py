# -*- coding: utf-8 -*-
"""
06_subtasks_appendix.py  (FINAL: no prompt-level scan)
Builds appendix tables/figures for Q13–Q17 using ONLY Sdon_clean.csv and Llogs_clean.csv.

Outputs:
  results/tab/T6_6_subtasks_Q13.csv ... Q17.csv
  results/tab/T6_6_subtasks_full.csv
  results/tab/T6_6_subtasks_columns_found.csv
  results/fig/F6_A_subtasks_Q13_deltap.png ... Q17_deltap.png
"""

import os, re, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE     = os.getcwd()
DERIVED  = os.path.join(BASE, "results", "derived")
FIG_DIR  = os.path.join(BASE, "results", "fig")
TAB_DIR  = os.path.join(BASE, "results", "tab")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

def wilson_ci(k, n):
    if n == 0: return (np.nan, np.nan, np.nan)
    z = 1.959963984540054
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2/(2*n)) / den
    half = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / den
    return float(p), float(max(0, centre-half)), float(min(1, centre+half))

def newcombe_diff_ci(k1,n1,k2,n2):
    _, l1, u1 = wilson_ci(k1, n1)
    _, l2, u2 = wilson_ci(k2, n2)
    p1 = k1/n1 if n1>0 else np.nan
    p2 = k2/n2 if n2>0 else np.nan
    return float(p1-p2), float(l1-u2), float(u1-l2)

def two_prop_z(k1,n1,k2,n2):
    if n1==0 or n2==0: return np.nan
    p1, p2 = k1/n1, k2/n2
    p = (k1+k2)/(n1+n2)
    se = math.sqrt(p*(1-p)*(1/n1 + 1/n2))
    if se == 0: return np.nan
    z = (p1-p2)/se
    return 2.0*(1.0 - 0.5*(1.0 + math.erf(abs(z)/math.sqrt(2))))

def fdr_bh(pvals, q=0.10):
    p = np.asarray(pvals, dtype=float)
    m = np.sum(np.isfinite(p))
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(p)+1)
    qvals = p * m / ranks
    q_sorted = np.minimum.accumulate(qvals[order][::-1])[::-1]
    out = np.full_like(p, np.nan, dtype=float); out[order] = q_sorted
    return out

def pretty_subtask(col: str) -> str:
    s = re.sub(r"^q1[3-7][^A-Za-z0-9]+", "", col, flags=re.IGNORECASE)
    s = re.sub(r"_sel$", "", s, flags=re.IGNORECASE)
    s = s.replace("_", " ").replace("  ", " ")
    s = s.replace("role play", "role‑play").replace("proof reading", "proof‑reading").replace("q a", "Q&A")
    s = s.strip()
    return s[0:1].upper() + s[1:] if s else col

def find_subtask_cols(df: pd.DataFrame, qtag: str):
    cols = []
    pat = re.compile(rf"^{re.escape(qtag)}[^A-Za-z0-9]+", flags=re.IGNORECASE)
    for c in df.columns:
        low = c.lower()
        if not pat.search(low): continue
        if "not_chosen" in low or "not chosen" in low or "breadth" in low: continue
        if low.startswith("q12__"): continue
        vals = pd.to_numeric(df[c], errors="coerce")
        uniq = np.unique(vals[~np.isnan(vals)])
        if uniq.size and np.all(np.isin(uniq, [0,1])):
            cols.append(c)
    return sorted(cols)

def get_binary_series(df: pd.DataFrame, col: str):
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return s
    else:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)

Q12_FAMS = {
    "Q13": ("q13", "q12__writing_and_professional_communication", "Q13 (Writing subtasks)"),
    "Q14": ("q14", "q12__brainstorming_and_personal_ideas_fun",   "Q14 (Brainstorming/fun subtasks)"),
    "Q15": ("q15", "q12__coding_programming_help",                "Q15 (Coding subtasks)"),
    "Q16": ("q16", "q12__language_practice_or_translation",       "Q16 (Language/translation subtasks)"),
    "Q17": ("q17", "q12__study_revision_or_exam_prep",            "Q17 (Study/exam subtasks)"),
}

def savefig_always(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200)
    print(f"[saved] {path}")

def main():
    sdon_path = os.path.join(DERIVED, "Sdon_clean.csv")
    llog_path = os.path.join(DERIVED, "Llogs_clean.csv")
    if not (os.path.exists(sdon_path) and os.path.exists(llog_path)):
        raise FileNotFoundError("Missing Sdon_clean.csv or Llogs_clean.csv in results/derived/")

    Sdon = pd.read_csv(sdon_path, dtype=str, keep_default_na=False)
    Llogs = pd.read_csv(llog_path, dtype=str, keep_default_na=False)

    columns_found_log = []
    all_rows = []

    for tag, (qtag, parent_col, fam_title) in Q12_FAMS.items():
        subs_sdon  = find_subtask_cols(Sdon,  qtag)
        subs_llogs = find_subtask_cols(Llogs, qtag)
        subs_union = sorted(list(set(subs_sdon) | set(subs_llogs)))

        columns_found_log.append({
            "family": tag,
            "n_cols_Sdon": len(subs_sdon),
            "n_cols_Llogs": len(subs_llogs),
            "n_cols_union": len(subs_union),
            "cols_Sdon": "; ".join(subs_sdon),
            "cols_Llogs": "; ".join(subs_llogs),
            "cols_union": "; ".join(subs_union),
        })
        print(f"[{tag}] found Sdon:{len(subs_sdon)} Llogs:{len(subs_llogs)} union:{len(subs_union)}")

        out_csv = os.path.join(TAB_DIR, f"T6_6_subtasks_{tag}.csv")
        out_fig = os.path.join(FIG_DIR, f"F6_A_subtasks_{tag}_deltap.png")

        if len(subs_union) == 0:
            pd.DataFrame(columns=[
                "family","subtask","p_Sdon","l_Sdon","u_Sdon","p_Llogs","l_Llogs","u_Llogs",
                "deltap","l95","u95","p","q_FDR","support_Sdon","support_Llogs"
            ]).to_csv(out_csv, index=False)

            plt.figure(figsize=(8, 3))
            plt.text(0.5, 0.5, f"No subtask columns detected for {tag}\n(See T6_6_subtasks_columns_found.csv)",
                     ha='center', va='center')
            plt.axis("off")
            savefig_always(out_fig); plt.close()
            continue

        # gating per cohort
        gate_sdon = pd.to_numeric(Sdon.get(parent_col, 0), errors="coerce").fillna(0).astype(int)
        gate_llog = pd.to_numeric(Llogs.get(parent_col,0), errors="coerce").fillna(0).astype(int)
        Sdon_g  = Sdon.loc[gate_sdon==1].copy()
        Llogs_g = Llogs.loc[gate_llog==1].copy()
        n1 = int(Sdon_g.shape[0]); n2 = int(Llogs_g.shape[0])

        fam_rows = []
        for sub in subs_union:
            s1 = get_binary_series(Sdon_g,  sub)
            s2 = get_binary_series(Llogs_g, sub)
            k1 = int(s1.sum()); k2 = int(s2.sum())

            p1, l1, u1 = wilson_ci(k1, n1)
            p2, l2, u2 = wilson_ci(k2, n2)
            dp, dl, du = newcombe_diff_ci(k1,n1,k2,n2)
            pval = two_prop_z(k1,n1,k2,n2)

            fam_rows.append({
                "family": fam_title,
                "subtask": pretty_subtask(sub),
                "p_Sdon": p1, "l_Sdon": l1, "u_Sdon": u1,
                "p_Llogs": p2, "l_Llogs": l2, "u_Llogs": u2,
                "deltap": dp, "l95": dl, "u95": du, "p": pval,
                "support_Sdon": f"{k1}/{n1}", "support_Llogs": f"{k2}/{n2}"
            })

        df_fam = pd.DataFrame(fam_rows)
        df_fam["q_FDR"] = fdr_bh(df_fam["p"].values, q=0.10)
        df_fam.to_csv(out_csv, index=False)
        all_rows.append(df_fam)

        # Δp-forest
        df_plot = df_fam.copy()
        order = np.argsort(-np.abs(df_plot["deltap"].values))
        df_plot = df_plot.iloc[order].reset_index(drop=True)

        y = np.arange(len(df_plot))
        d = (df_plot["deltap"].values * 100).astype(float)
        l = (df_plot["l95"].values   * 100).astype(float)
        u = (df_plot["u95"].values   * 100).astype(float)
        labels = df_plot["subtask"].astype(str).tolist()

        plt.figure(figsize=(max(9, 0.55*len(labels)+6), max(4.8, 0.45*len(labels)+1.2)))
        plt.axvline(0, linewidth=1)

        any_plotted = False
        for i,(di,li,ui) in enumerate(zip(d,l,u)):
            if np.isfinite(di) and np.isfinite(li) and np.isfinite(ui):
                plt.errorbar(di, i, xerr=[[di-li],[ui-di]], fmt='o', capsize=3)
                shift = 1 if ui >= 0 else -1
                plt.text(ui + shift, i, f"{di:.1f} pp", va='center', fontsize=9)
                any_plotted = True
            if df_plot.loc[i, "q_FDR"] < 0.10:
                labels[i] = labels[i] + " †"

        plt.yticks(y, labels)
        plt.xlabel("Δ percentage points (Sdon − Llogs)")
        title = f"{fam_title}: subtask differences (indicative)\n" \
                f"Gated on parent family; † FDR-BH q<0.10; n={n1} vs n={n2}"
        if not any_plotted:
            title += "\n(Only non-finite CIs; counts are extremely small — see table for supports.)"
        plt.title(title)
        plt.tight_layout()
        savefig_always(out_fig); plt.close()

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(os.path.join(TAB_DIR, "T6_6_subtasks_full.csv"), index=False)
    pd.DataFrame(columns_found_log).to_csv(os.path.join(TAB_DIR, "T6_6_subtasks_columns_found.csv"), index=False)

    print("Subtask appendix: tables and figures written (no prompt-level dependency).")

if __name__ == "__main__":
    main()
