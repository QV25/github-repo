# -*- coding: utf-8 -*-
"""
03_sq1_survey_landscape.py — SQ1 visuals, tables & LaTeX (Q-labels, top-10 coefs, dim summary)

INPUT  (results/derived/):
  - S_clean.csv
  - Sdon_clean.csv

OUTPUT (key):
  - results/fig/F6_1_association_heatmap.png / .pdf
  - results/tab/T6_1_association_matrix.csv
  - results/tab/T6_1_jaccard_q12.csv
  - results/fig/F6_2_scree.png
  - results/tab/T6_2_parallel_analysis.csv
  - results/tab/T6_2_loadings.csv
  - results/tab/T6_2_loadings.tex
  - results/tab/T6_2_dim_summary.txt          <-- NEW
  - results/derived/S_factor_scores.csv
  - results/derived/Sdon_factor_scores.csv
  - results/tab/T6_3_coefficients_SQ1_dimX_score.csv  (+ partial R²)
  - results/tab/T6_3_coefficients_SQ1_dimX_score.tex  (Q-labelled terms)
  - results/fig/F6_3_coefficients_SQ1_dimX_full.png
  - results/fig/F6_3_coefficients_SQ1_dimX_top10.png
"""
import os, re, random, unicodedata, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional libs
try:
    import statsmodels.api as sm
    from statsmodels.tools.tools import add_constant
except Exception:
    sm = None

try:
    from sklearn.decomposition import PCA
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

try:
    from factor_analyzer.rotator import Rotator
    HAVE_FACTOR_ANALYZER = True
except Exception:
    HAVE_FACTOR_ANALYZER = False

warnings.filterwarnings("ignore")

BASE = os.getcwd()
DERIVED_DIR = os.path.join(BASE, "results", "derived")
FIG_DIR = os.path.join(BASE, "results", "fig")
TAB_DIR = os.path.join(BASE, "results", "tab")
os.makedirs(DERIVED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)

# ---------- helpers ----------
def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = s.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip()

def latex_escape(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}")
             .replace("&", "\\&").replace("%", "\\%")
             .replace("$", "\\$").replace("#", "\\#")
             .replace("_", "\\_").replace("{", "\\{")
             .replace("}", "\\}").replace("~", "\\textasciitilde{}")
             .replace("^", "\\textasciicircum{}"))

def write_latex_table(df: pd.DataFrame, path: str, column_format: str|None=None,
                      index: bool=True, header: bool=True, booktabs: bool=True,
                      escape_index: bool=True, escape_cells: bool=False):
    cols = df.columns.tolist()
    ncols = len(cols) + (1 if index else 0)
    if column_format is None:
        column_format = "l" + "r"*(ncols-1)
    lines = []
    lines.append("\\begin{tabular}{%s}" % column_format)
    if booktabs: lines.append("\\toprule")
    if header:
        hdr = []
        if index: hdr.append("")
        for c in cols:
            hdr.append(latex_escape(str(c)))
        lines.append(" & ".join(hdr) + " \\\\")
        lines.append("\\midrule" if booktabs else "\\hline")
    for r in range(df.shape[0]):
        row = []
        if index:
            row.append(latex_escape(str(df.index[r])) if escape_index else str(df.index[r]))
        for c in cols:
            val = df.iloc[r][c]
            s = "" if pd.isna(val) else str(val)
            row.append(latex_escape(s) if escape_cells else s)
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule" if booktabs else "\\hline")
    lines.append("\\end{tabular}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def rankdata(a):
    a = np.asarray(a, dtype=float); order = a.argsort()
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(len(a), dtype=float)
    uniq, first = np.unique(a[order], return_index=True)
    for i in range(len(uniq)):
        start = first[i]; end = first[i+1] if i+1 < len(uniq) else len(a)
        ranks[start:end] = (start + end - 1) / 2.0
    return ranks + 1.0

def pearson_corr(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y); x = x[m]; y = y[m]
    if len(x) < 3: return np.nan
    x = x - x.mean(); y = y - y.mean()
    denom = (np.sqrt((x**2).sum()) * np.sqrt((y**2).sum()))
    if denom == 0: return np.nan
    return float((x*y).sum()/denom)

def spearman_rho(x, y):
    xs = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy()
    ys = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
    m = np.isfinite(xs) & np.isfinite(ys); xs = xs[m]; ys = ys[m]
    if len(xs) < 3: return np.nan
    return pearson_corr(rankdata(xs), rankdata(ys))

def cramers_v_from_table(tab):
    tab = np.asarray(tab, dtype=float)
    n = tab.sum()
    if n == 0: return np.nan
    exp = tab.sum(1, keepdims=True).dot(tab.sum(0, keepdims=True)) / n
    with np.errstate(invalid='ignore', divide='ignore'):
        chi2 = np.nansum((tab - exp)**2 / exp)
    r, c = tab.shape; denom = n * (min(r-1, c-1))
    if denom <= 0: return np.nan
    return float(np.sqrt(chi2/denom))

def correlation_ratio(cats, vals):
    cats = np.asarray(cats)
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy()
    m = np.isfinite(vals) & pd.Series(cats).notna().values
    cats = cats[m]; vals = vals[m]
    if len(vals) < 3: return np.nan
    levels = pd.unique(cats); gm = vals.mean()
    ssb = 0.0; sst = ((vals - gm)**2).sum()
    for lv in levels:
        g = vals[cats == lv]
        if len(g) == 0: continue
        ssb += len(g) * (g.mean() - gm)**2
    if sst == 0: return np.nan
    return float(np.sqrt(ssb/sst))

def jaccard_binary(u, v):
    u = (pd.to_numeric(pd.Series(u), errors="coerce").fillna(0).to_numpy() > 0).astype(int)
    v = (pd.to_numeric(pd.Series(v), errors="coerce").fillna(0).to_numpy() > 0).astype(int)
    inter = int(np.logical_and(u==1, v==1).sum()); union = int(np.logical_or(u==1, v==1).sum())
    if union == 0: return np.nan
    return inter/union

def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    p, k = Phi.shape; R = np.eye(k); d = 0
    for _ in range(q):
        d_old = d; L = Phi.dot(R)
        u, s, vh = np.linalg.svd(Phi.T.dot(L**3 - (gamma/p)*L.dot(np.diag(np.diag(L.T.dot(L))))))
        R = u.dot(vh); d = np.sum(s)
        if d_old != 0 and d/d_old < 1+tol: break
    return Phi.dot(R)

def parallel_analysis(data, reps=200, seed=2025):
    rng = np.random.default_rng(seed)
    n, p = data.shape
    Z = (data - np.nanmean(data, 0)) / np.nanstd(data, 0, ddof=0)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    R = np.corrcoef(Z, rowvar=False)
    evals, _ = np.linalg.eig(R); evals = np.sort(np.real(evals))[::-1]
    rand = np.zeros((reps, p))
    for i in range(reps):
        Xr = rng.standard_normal(size=(n, p))
        Rr = np.corrcoef(Xr, rowvar=False)
        vals, _ = np.linalg.eig(Rr)
        rand[i,:] = np.sort(np.real(vals))[::-1]
    return pd.DataFrame({"component": np.arange(1, p+1), "eigenvalue": evals, "mean_rand": rand.mean(0)})

def map_q18_mid(x: str) -> float:
    x = normalize_text(x)
    if not x: return np.nan
    m = re.match(r"^(\d{1,3})\s*-\s*(\d{1,3})", x)
    if m: return (float(m.group(1)) + float(m.group(2))) / 2.0
    m2 = re.match(r"^(\d{1,3})", x)
    if m2: return float(m2.group(1))
    return np.nan

# ---- autodetect Q1..Q6 -> roles + pretty labels ----
TOKENS = {
    "status": ["Student","Working","Both","Other"],
    "age_band": ["< 18","18-24","25-34","35-44","45+"],
    "gender": ["Man","Woman","Prefer not to say","Non-binary"],
    "field": ["STEM","Humanities","Business","Creative","media","economics","social sciences"],
    "plan": ["Free","Plus","Pro"],
    "device": ["Laptop / desktop","Smartphone","Mixed equally"],
}
ROLE_NAME = {"status":"Status","age_band":"Age","gender":"Gender","field":"Field","plan":"Plan","device":"Device"}

def detect_qmap(S):
    candidates = {f"Q{i}": set(map(normalize_text, S[f"Q{i}"].dropna().astype(str).unique().tolist()))
                  for i in range(1,7) if f"Q{i}" in S.columns}
    out = {}; used = set()
    for role, keys in TOKENS.items():
        best_q, best_hit = None, 0
        for q, vals in candidates.items():
            if q in used: continue
            hit = sum(1 for k in keys for v in vals if k in v)
            if hit > best_hit:
                best_q, best_hit = q, hit
        if best_q:
            out[role] = best_q; used.add(best_q)
    return out

def pretty_dummy_with_q(term, qmap):
    # term like "Q5_Plus"
    if term == "const": return "const"
    if "_" not in term: return term
    q, lvl = term.split("_", 1)
    role = next((r for r,qx in qmap.items() if qx == q), None)
    role_name = ROLE_NAME.get(role, q)
    return f"{q} ({role_name}): {lvl.replace('_',' ')}"

# ---------- main ----------
def run_sq1():
    S_PATH = os.path.join(DERIVED_DIR, "S_clean.csv")
    SDON_PATH = os.path.join(DERIVED_DIR, "Sdon_clean.csv")
    if not os.path.exists(S_PATH):
        raise FileNotFoundError(f"Missing {S_PATH}. Run the cleaner first.")

    # --- load S ---
    S = pd.read_csv(S_PATH, dtype=str, keep_default_na=False)
    for col in ["Q7_mid","Q8_mid","Q9_mid","Q11_score","task_breadth_main",
                "subtask_breadth_wri","subtask_breadth_bra","subtask_breadth_cod",
                "subtask_breadth_lan","subtask_breadth_stu"]:
        if col in S.columns: S[col] = pd.to_numeric(S[col], errors="coerce")

    if "usage_index_survey" not in S.columns:
        Z = S[["Q7_mid","Q8_mid","Q9_mid"]].apply(lambda s: (s - s.mean())/s.std(ddof=0))
        S["usage_index_survey"] = Z.mean(axis=1)
    else:
        S["usage_index_survey"] = pd.to_numeric(S["usage_index_survey"], errors="coerce")

    if "Q18" in S.columns: S["Q18_mid"] = S["Q18"].apply(map_q18_mid)

    QMAP_DET = detect_qmap(S)

    # ---------- association matrix ----------
    NUMERIC_COLS = [c for c in ["Q7_mid","Q8_mid","Q9_mid","Q11_score","task_breadth_main",
                                "subtask_breadth_wri","subtask_breadth_bra","subtask_breadth_cod",
                                "subtask_breadth_lan","subtask_breadth_stu","usage_index_survey","Q18_mid"]
                    if c in S.columns]
    for c in NUMERIC_COLS: S[c] = pd.to_numeric(S[c], errors="coerce")

    CATEGORICAL_COLS = [q for q in [QMAP_DET.get(k) for k in ["plan","device","status","field","age_band","gender"]] if q in S.columns]
    if "Q10" in S.columns: CATEGORICAL_COLS.append("Q10")
    if "Q11_band" in S.columns: CATEGORICAL_COLS.append("Q11_band")

    Q12_BIN = [
        "q12__writing_and_professional_communication",
        "q12__brainstorming_and_personal_ideas_fun",
        "q12__coding_programming_help",
        "q12__language_practice_or_translation",
        "q12__study_revision_or_exam_prep",
        "q12__other",
    ]
    Q12_PRESENT = [c for c in Q12_BIN if c in S.columns]
    BINARY_COLS = Q12_PRESENT

    VARS = NUMERIC_COLS + CATEGORICAL_COLS + BINARY_COLS
    M = np.zeros((len(VARS), len(VARS)), dtype=float)
    for i, vi in enumerate(VARS):
        for j, vj in enumerate(VARS):
            if i == j:
                M[i,j] = 1.0; continue
            xi, xj = S[vi], S[vj]
            if (vi in NUMERIC_COLS) and (vj in NUMERIC_COLS):
                val = abs(spearman_rho(xi, xj))
            elif (vi in CATEGORICAL_COLS + BINARY_COLS) and (vj in CATEGORICAL_COLS + BINARY_COLS):
                tab = pd.crosstab(xi, xj).values; val = cramers_v_from_table(tab)
            else:
                if vi in NUMERIC_COLS:
                    val = correlation_ratio(xj.values, xi.values)
                else:
                    val = correlation_ratio(xi.values, xj.values)
            M[i,j] = val if np.isfinite(val) else np.nan

    pd.DataFrame(M, index=VARS, columns=VARS).to_csv(os.path.join(TAB_DIR, "T6_1_association_matrix.csv"), index=True)
    for ext in ("png","pdf"):
        plt.figure(figsize=(12,10))
        im = plt.imshow(M, aspect='auto', interpolation='nearest')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(np.arange(len(VARS)), VARS, rotation=90); plt.yticks(np.arange(len(VARS)), VARS)
        plt.title("Association heatmap (|Spearman| / Cramér's V / η)")
        plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, f"F6_1_association_heatmap.{ext}"), dpi=200); plt.close()

    if len(Q12_PRESENT) >= 2:
        J = np.zeros((len(Q12_PRESENT), len(Q12_PRESENT)), dtype=float)
        for i, ci in enumerate(Q12_PRESENT):
            for j, cj in enumerate(Q12_PRESENT):
                J[i,j] = 1.0 if i==j else jaccard_binary(S[ci], S[cj])
        pd.DataFrame(J, index=Q12_PRESENT, columns=Q12_PRESENT).to_csv(os.path.join(TAB_DIR, "T6_1_jaccard_q12.csv"))

    # ---------- PCA items ----------
    PCA_ITEMS = [c for c in ["usage_index_survey","Q11_score","task_breadth_main",
                             "subtask_breadth_wri","subtask_breadth_bra","subtask_breadth_cod",
                             "subtask_breadth_lan","Q18_mid"] if c in S.columns]
    if len(PCA_ITEMS) < 2:
        raise ValueError(f"Too few PCA items available in S: {PCA_ITEMS}")

    X = S[PCA_ITEMS].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(how="all", axis=0)
    X_std = (X - X.mean(0)) / X.std(0, ddof=0)
    X_std = X_std.fillna(0.0)

    pa = parallel_analysis(X_std.values, reps=200, seed=RANDOM_SEED)
    pa.to_csv(os.path.join(TAB_DIR, "T6_2_parallel_analysis.csv"), index=False)
    k = int((pa["eigenvalue"] > pa["mean_rand"]).sum()); k = max(1, min(k, X_std.shape[1]))

    if HAVE_SKLEARN:
        pca = PCA(n_components=k, random_state=RANDOM_SEED)
        comps = pca.fit_transform(X_std.values); loadings = pca.components_.T
    else:
        R = np.corrcoef(X_std.values, rowvar=False)
        evals, evecs = np.linalg.eig(R); idx = np.argsort(evals)[::-1]
        loadings = np.real(evecs[:, idx])[:, :k]; comps = X_std.values.dot(loadings)

    load_rot = Rotator(method='oblimin').fit_transform(loadings) if HAVE_FACTOR_ANALYZER else varimax(loadings)
    scores = X_std.values.dot(load_rot)

    load_df = pd.DataFrame(load_rot, index=PCA_ITEMS, columns=[f"Dim{i+1}" for i in range(load_rot.shape[1])])
    load_df.to_csv(os.path.join(TAB_DIR, "T6_2_loadings.csv"))

    # LaTeX loadings with bold >= .30
    L = load_df.copy()
    def fmt_bold(x): 
        return f"\\textbf{{{x:.2f}}}" if abs(x) >= 0.30 else f"{x:.2f}"
    L = L.applymap(fmt_bold)
    L.index = [latex_escape(str(i)) for i in L.index]
    write_latex_table(L, os.path.join(TAB_DIR, "T6_2_loadings.tex"),
                      column_format="l" + "r"*L.shape[1],
                      index=True, header=True, booktabs=True,
                      escape_index=False, escape_cells=False)

    # Scree (English labels) + vertical line at k
    plt.figure(figsize=(6,4))
    plt.plot(pa["component"], pa["eigenvalue"], marker="o", label="Observed")
    plt.plot(pa["component"], pa["mean_rand"], marker="x", label="Parallel mean")
    plt.axvline(k, linestyle="--")
    plt.xlabel("Component"); plt.ylabel("Eigenvalue (λ)")
    plt.title("Scree & Parallel Analysis")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "F6_2_scree.png"), dpi=200); plt.close()

    # Dim-summary (top loadings with signs + suggested label)
    def suggest_label(top_items):
        names = [n for n,_ in top_items]
        if "usage_index_survey" in names and "task_breadth_main" in names:
            return "Intensity & portfolio breadth"
        if "Q18_mid" in names:
            return "Study/work orientation"
        if "Q11_score" in names:
            return "Prompt length / input form"
        return "Miscellaneous"
    with open(os.path.join(TAB_DIR, "T6_2_dim_summary.txt"), "w") as f:
        for j in range(load_df.shape[1]):
            col = load_df.iloc[:, j]
            top = col.reindex(col.abs().sort_values(ascending=False).index)[:5]
            items = [(idx, float(top[idx])) for idx in top.index]
            label = suggest_label(items[:3])
            f.write(f"Dimension {j+1} — suggested label: {label}\n")
            for n,v in items:
                sign = "+" if v>=0 else "−"
                f.write(f"  {n}: {sign}{abs(v):.2f}\n")
            f.write("\n")

    # Factor scores back to S + to Sdon
    S_scores = S.loc[X_std.index, :].copy()
    for i in range(load_df.shape[1]): S_scores[f"SQ1_dim{i+1}_score"] = scores[:, i]
    S_scores.to_csv(os.path.join(DERIVED_DIR, "S_factor_scores.csv"), index=False)

    if os.path.exists(SDON_PATH):
        Sdon = pd.read_csv(SDON_PATH, dtype=str, keep_default_na=False)
        for c in ["Q7_mid","Q8_mid","Q9_mid"]:
            Sdon[c] = pd.to_numeric(Sdon[c], errors="coerce")
        if "usage_index_survey" not in Sdon.columns:
            mu = S[["Q7_mid","Q8_mid","Q9_mid"]].mean(numeric_only=True)
            sd = S[["Q7_mid","Q8_mid","Q9_mid"]].std(ddof=0, numeric_only=True)
            Zs = (Sdon[["Q7_mid","Q8_mid","Q9_mid"]] - mu) / sd
            Sdon["usage_index_survey"] = Zs.mean(axis=1)
        else:
            Sdon["usage_index_survey"] = pd.to_numeric(Sdon["usage_index_survey"], errors="coerce")
        if ("Q18_mid" not in Sdon.columns) and ("Q18" in Sdon.columns):
            Sdon["Q18_mid"] = Sdon["Q18"].apply(map_q18_mid)
        for col in PCA_ITEMS:
            if col not in Sdon.columns: Sdon[col] = np.nan
            Sdon[col] = pd.to_numeric(Sdon[col], errors="coerce")
        muX = X.mean(0); sdX = X.std(0, ddof=0)
        Xs = Sdon[PCA_ITEMS].copy()
        Xs_std = (Xs - muX) / sdX
        Xs_std = Xs_std.fillna(0.0)
        Sc = Xs_std.values.dot(load_df.values)
        for i in range(load_df.shape[1]): Sdon[f"SQ1_dim{i+1}_score"] = Sc[:, i]
        Sdon.to_csv(os.path.join(DERIVED_DIR, "Sdon_factor_scores.csv"), index=False)

    # ---------- OLS HC3 on factor scores (survey-only) ----------
    if sm is not None:
        predictors = [q for q in [QMAP_DET.get(k) for k in ["plan","device","status","field","age_band","gender"]] if q in S_scores.columns]
        mod = S_scores[predictors + [c for c in S_scores.columns if c.startswith("SQ1_dim")]].dropna()
        Xmod = pd.get_dummies(mod[predictors], drop_first=True)
        Xmod = add_constant(Xmod, has_constant="add")

        for sc in [c for c in mod.columns if c.startswith("SQ1_dim")]:
            y = (mod[sc] - mod[sc].mean())/mod[sc].std(ddof=0)
            fit = sm.OLS(y.astype(float), Xmod.astype(float)).fit(cov_type="HC3")
            coefs = fit.params; ses = fit.bse; pvals = fit.pvalues
            tvals = coefs / ses
            df_res = int(fit.df_resid)
            partial_r2 = (tvals**2) / (tvals**2 + df_res)

            out = pd.DataFrame({
                "term": coefs.index,
                "beta": coefs.values,
                "se": ses.values,
                "l95": coefs.values - 1.96*ses.values,
                "u95": coefs.values + 1.96*ses.values,
                "p": pvals.values,
                "partial_R2": partial_r2.values
            })
            out["pretty"] = out["term"].apply(lambda t: pretty_dummy_with_q(t, QMAP_DET))
            out.to_csv(os.path.join(TAB_DIR, f"T6_3_coefficients_{sc}.csv"), index=False)

            # LaTeX export (Q labels)
            def star(p): 
                return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
            L = out.copy()
            L["Estimate [95% CI]"] = L.apply(lambda r: f"{r['beta']:.2f} [{r['l95']:.2f}, {r['u95']:.2f}] {star(r['p'])}", axis=1)
            L["Partial R$^2$"] = L["partial_R2"].apply(lambda x: f"{x:.03f}")
            L = L[["pretty","Estimate [95% CI]","Partial R$^2$"]].rename(columns={"pretty":"Term"})
            L["Term"] = L["Term"].apply(latex_escape)
            write_latex_table(L, os.path.join(TAB_DIR, f"T6_3_coefficients_{sc}.tex"),
                              column_format="l r r", index=False, header=True, booktabs=True,
                              escape_index=False, escape_cells=False)

            # FULL plot
            terms = out["pretty"].tolist()
            betas = out["beta"].values; l95 = out["l95"].values; u95 = out["u95"].values
            xi = np.arange(len(terms))
            plt.figure(figsize=(max(8, len(terms)*0.5), 4))
            plt.axhline(0, linewidth=1)
            plt.errorbar(xi, betas, yerr=[betas - l95, u95 - betas], fmt='o', capsize=3)
            plt.xticks(xi, terms, rotation=90)
            plt.ylabel("Standardized beta (HC3 95% CI)")
            plt.title(f"Coefficients: {sc}")
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"F6_3_coefficients_{sc}_full.png"), dpi=200)
            plt.close()

            # TOP-10 plot (exclude const), horizontal, symmetric x
            mask = out["term"] != "const"
            out2 = out[mask].copy()
            out2["absb"] = out2["beta"].abs()
            out2 = out2.sort_values("absb", ascending=False).head(10).sort_values("beta")
            ytick = np.arange(out2.shape[0])
            plt.figure(figsize=(10, 0.5*max(6, out2.shape[0])+1))
            plt.axvline(0, linewidth=1)
            plt.errorbar(out2["beta"], ytick,
                         xerr=[out2["beta"]-out2["l95"], out2["u95"]-out2["beta"]],
                         fmt='o', capsize=3)
            plt.yticks(ytick, out2["pretty"])
            xmax = float(np.ceil((np.nanmax(np.abs(out2[["l95","u95"]].values))*1.10)*10)/10)
            plt.xlim(-xmax, xmax)
            plt.xlabel("Standardized beta (HC3 95% CI)")
            plt.title(f"Top-10 coefficients: {sc}")
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"F6_3_coefficients_{sc}_top10.png"), dpi=200)
            plt.close()

    print("SQ1 pipeline completed.")

if __name__ == "__main__":
    run_sq1()
