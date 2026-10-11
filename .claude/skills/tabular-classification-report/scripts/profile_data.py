"""Stage 1 - deterministic data profile.

Computes facts about a tabular classification dataset and raises *flags*
(potential data-quality issues). It never changes the data and never decides
anything: the plan written in Stage 2 decides what to do about each flag.

Usage:
    python profile_data.py --data train.csv [--test test.csv] [--target label] --out runs/x
Writes  runs/x/profile.json  (full facts)  and  runs/x/profile.md  (digest for the LLM).
"""
from __future__ import annotations

import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from common import column_types, dump_json, load_dataset, split_xy

warnings.filterwarnings("ignore")

PLACEHOLDER_LEVELS = {"unknown", "other", "others", "none", "missing", "n/a", "na", "-", "?", "unspecified"}
RARE_SHARE = 0.01          # a level below 1% of rows is "rare"
LEAK_SCORE = 0.97          # single-feature AUC above this is suspicious
EXACT_R2 = 0.999           # numeric column almost perfectly reconstructable from the others
HIGH_CORR = 0.8
DRIFT_KS = 0.1
DRIFT_TVD = 0.1


def _r(x, k=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


def numeric_stats(s: pd.Series) -> dict:
    v = s.dropna()
    q1, q3 = v.quantile([0.25, 0.75])
    iqr = q3 - q1
    zeros = int((v == 0).sum())
    nonzero = v[v != 0]
    d = {
        "missing_train": int(s.isna().sum()), "n_unique": int(v.nunique()),
        "mean": _r(v.mean()), "std": _r(v.std()), "min": _r(v.min()), "q1": _r(q1),
        "median": _r(v.median()), "q3": _r(q3), "max": _r(v.max()), "skew": _r(v.skew(), 3),
        "zeros": zeros, "zero_share": _r(zeros / len(v)),
        "outlier_share_1.5iqr": _r(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).mean()),
        "outlier_share_3iqr": _r(((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)).mean()),
        "integer_valued": bool(np.allclose(v, np.round(v))),
    }
    if zeros and len(nonzero):
        nq1, nq3 = nonzero.quantile([0.25, 0.75])
        d["zero_isolated"] = bool(0 < nq1 - 1.5 * (nq3 - nq1))  # 0 sits far below the bulk
        d["min_nonzero"] = _r(nonzero.abs().min())
    return d


def categorical_stats(s: pd.Series, test: pd.Series | None) -> dict:
    vc = s.value_counts(dropna=True)
    share = vc / vc.sum()
    rare = share[share < RARE_SHARE]
    d = {
        "missing_train": int(s.isna().sum()), "n_levels": int(len(vc)),
        "top_levels": {str(k): int(v) for k, v in vc.head(6).items()},
        "dominant_share": _r(share.iloc[0]) if len(share) else None,
        "rare_levels": int(len(rare)), "rare_share_total": _r(rare.sum()),
        "placeholder_levels": [str(k) for k in vc.index if str(k).strip().lower() in PLACEHOLDER_LEVELS],
    }
    if test is not None:
        unseen = set(test.dropna().unique()) - set(vc.index)
        d["unseen_test_levels"] = sorted(map(str, unseen))
        d["unseen_test_rows"] = int(test.isin(unseen).sum())
    return d


def single_feature_power(X: pd.DataFrame, y: pd.Series, seed: int) -> dict:
    """CV ROC-AUC of a depth-4 tree on each feature alone (macro OvR for multiclass)."""
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    classes = sorted(y.unique())
    out = {}
    for c in X.columns:
        col = X[c]
        if pd.api.types.is_numeric_dtype(col):
            f = col.fillna(col.min() - 1 if col.notna().any() else 0).to_numpy().reshape(-1, 1)
        else:
            f = col.astype(str).astype("category").cat.codes.to_numpy().reshape(-1, 1)
        proba = cross_val_predict(DecisionTreeClassifier(max_depth=4, random_state=seed),
                                  f, y, cv=cv, method="predict_proba")
        if len(classes) == 2:
            out[c] = _r(roc_auc_score((y == classes[1]).astype(int), proba[:, 1]), 3)
        else:
            out[c] = _r(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes), 3)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def linear_redundancy(X: pd.DataFrame, num: list[str], seed: int) -> dict:
    """R^2 of each numeric column regressed on the other numerics; formula when ~exact."""
    if len(num) < 2:
        return {}
    d = X[num].dropna()
    if len(d) > 20000:
        d = d.sample(20000, random_state=seed)
    out = {}
    for c in num:
        others = [o for o in num if o != c]
        m = LinearRegression().fit(d[others], d[c])
        r2 = m.score(d[others], d[c])
        entry = {"r2": _r(r2, 5)}
        if r2 >= EXACT_R2:
            # keep only terms that matter on the standardised scale
            std_coef = m.coef_ * d[others].std().to_numpy() / (d[c].std() or 1)
            terms = [(o, k) for o, k, s in zip(others, m.coef_, std_coef) if abs(s) > 0.01]
            m2 = LinearRegression().fit(d[[t for t, _ in terms]], d[c])
            entry["formula"] = " + ".join(f"{k:.4g}*{t}" for t, k in zip([t for t, _ in terms], m2.coef_))
            entry["formula_intercept"] = _r(m2.intercept_, 5)
            entry["formula_r2"] = _r(m2.score(d[[t for t, _ in terms]], d[c]), 7)
        out[c] = entry
    return out


def zero_blocks(X: pd.DataFrame, y: pd.Series, num: list[str]) -> list[dict]:
    """Groups of numeric columns that are zero on (almost) the same rows."""
    zsets = {c: set(np.flatnonzero((X[c] == 0).to_numpy())) for c in num}
    zsets = {c: s for c, s in zsets.items() if s}
    seen, blocks = set(), []
    for c in zsets:
        if c in seen:
            continue
        group = [c] + [o for o in zsets if o != c and o not in seen and
                       len(zsets[c] & zsets[o]) / len(zsets[c] | zsets[o]) >= 0.8]
        if len(group) < 2:
            continue
        seen.update(group)
        rows = sorted(set.intersection(*(zsets[g] for g in group)))
        dist = y.iloc[rows].value_counts(normalize=True).round(3).to_dict()
        blocks.append({"columns": group, "rows": len(rows), "target_dist_in_block": dist})
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--test")
    ap.add_argument("--target")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    ds = load_dataset(a.data, a.test, a.target, out, seed=a.seed)
    train, test, target = ds["train"], ds["test"], ds["target"]
    X, y = split_xy(train, target)
    Xt, yt = split_xy(test, target)
    num, cat = column_types(X)

    vc = y.value_counts()
    task = "binary" if len(vc) == 2 else "multiclass"
    prof = {
        "load": ds["log"], "task": task, "target": target,
        "n_train": len(train), "n_test": len(test), "n_features": X.shape[1],
        "numeric_columns": num, "categorical_columns": cat,
        "target_train": {str(k): int(v) for k, v in vc.items()},
        "target_test": {str(k): int(v) for k, v in yt.value_counts().items()},
        "minority_share": _r(vc.min() / vc.sum()), "imbalance_ratio": _r(vc.max() / vc.min(), 2),
        "onehot_width_estimate": len(num) + int(sum(X[c].nunique() for c in cat)),
    }
    prof["numeric"] = {c: numeric_stats(X[c]) for c in num}
    for c in num:
        prof["numeric"][c]["missing_test"] = int(Xt[c].isna().sum())
        ks = ks_2samp(X[c].dropna(), Xt[c].dropna())
        prof["numeric"][c]["drift_ks"] = _r(ks.statistic, 3)
        prof["numeric"][c]["drift_ks_p"] = _r(ks.pvalue, 4)
    prof["categorical"] = {c: categorical_stats(X[c], Xt[c]) for c in cat}
    for c in cat:
        p = X[c].value_counts(normalize=True)
        q = Xt[c].value_counts(normalize=True)
        prof["categorical"][c]["missing_test"] = int(Xt[c].isna().sum())
        prof["categorical"][c]["drift_tvd"] = _r(0.5 * p.subtract(q, fill_value=0).abs().sum(), 3)
        if task == "binary":
            pos = vc.idxmin()  # minority class is treated as the positive class
            rate = (y == pos).groupby(X[c]).mean()
            top = X[c].value_counts().head(8).index
            prof["categorical"][c]["positive_rate_by_level"] = {str(k): _r(rate[k], 3) for k in top}

    prof["missing"] = {
        "rows_with_any_missing_train": int(X.isna().any(axis=1).sum()),
        "rows_with_any_missing_test": int(Xt.isna().any(axis=1).sum()),
        "cells_missing_train": int(X.isna().sum().sum()),
    }
    feat_dups = X.duplicated(keep=False)
    prof["duplicates"] = {
        "full_rows": int(train.duplicated().sum()),
        "feature_dups_with_conflicting_label": int(
            train[feat_dups].groupby(list(X.columns), dropna=False)[target].nunique().gt(1).sum())
        if feat_dups.any() else 0,
        "test_rows_also_in_train": int(Xt.merge(X.drop_duplicates(), how="inner").shape[0]),
    }
    corr = X[num].corr() if len(num) > 1 else pd.DataFrame()
    prof["high_corr_pairs"] = [
        {"a": a_, "b": b_, "r": _r(corr.loc[a_, b_], 3)}
        for i, a_ in enumerate(num) for b_ in num[i + 1:] if abs(corr.loc[a_, b_]) >= HIGH_CORR]
    prof["linear_redundancy"] = linear_redundancy(X, num, a.seed)
    prof["zero_blocks"] = zero_blocks(X, y, num)
    prof["single_feature_auc"] = single_feature_power(X, y, a.seed)

    # ---------- flags: facts that call for an explicit decision in the plan ----------
    flags = []
    def flag(kind, column, evidence):
        flags.append({"kind": kind, "column": column, "evidence": evidence})

    if ds["log"]["train_rows_missing_label"] or ds["log"]["test_rows_missing_label"]:
        flag("missing_label", target, f"{ds['log']['train_rows_missing_label']} train / "
             f"{ds['log']['test_rows_missing_label']} test rows dropped (cannot be imputed)")
    if prof["minority_share"] < 0.35:
        flag("class_imbalance", target, f"minority share {prof['minority_share']}, ratio {prof['imbalance_ratio']}:1")
    miss = {c: (prof["numeric"].get(c) or prof["categorical"].get(c)) for c in num + cat}
    miss = {c: f"{i['missing_train']}/{i['missing_test']}" for c, i in miss.items()
            if i["missing_train"] or i["missing_test"]}
    if miss:
        worst = max(int(v.split("/")[0]) for v in miss.values()) / len(train)
        flag("missing_values", list(miss), f"train/test missing counts {miss}; worst column "
             f"{worst:.2%} of rows; rows with any missing: train {prof['missing']['rows_with_any_missing_train']}")
    for c, s in prof["numeric"].items():
        if s.get("zero_isolated") and s["zero_share"] >= 0.001:
            flag("isolated_zeros", c, f"{s['zeros']} zeros ({s['zero_share']:.2%}) far below the non-zero bulk")
        if s["outlier_share_3iqr"] >= 0.005 or abs(s["skew"] or 0) >= 2:
            flag("heavy_tail_or_outliers", c, f"skew {s['skew']}, {s['outlier_share_3iqr']:.2%} beyond 3*IQR, max {s['max']}")
        if s["integer_valued"] and s["n_unique"] <= 10:
            flag("numeric_low_cardinality", c, f"{s['n_unique']} distinct integer values - maybe categorical")
        if s["drift_ks"] >= DRIFT_KS and s["drift_ks_p"] < 0.01:  # large AND not explainable by sample size
            flag("train_test_drift", c, f"KS statistic {s['drift_ks']} (p={s['drift_ks_p']})")
    for b in prof["zero_blocks"]:
        flag("co_occurring_zeros", b["columns"], f"{b['rows']} rows zero in all; target distribution there {b['target_dist_in_block']}")
    for c, s in prof["categorical"].items():
        if s["rare_levels"]:
            flag("rare_levels", c, f"{s['rare_levels']} levels <1% each, {s['rare_share_total']:.2%} of rows in total")
        if s["dominant_share"] and s["dominant_share"] >= 0.9:
            flag("near_constant", c, f"top level covers {s['dominant_share']:.1%}")
        if s["placeholder_levels"]:
            flag("placeholder_levels", c, f"explicit placeholder values {s['placeholder_levels']}")
        if s["n_levels"] > 50:
            flag("high_cardinality", c, f"{s['n_levels']} levels")
        if s.get("unseen_test_rows"):
            flag("unseen_test_levels", c, f"{s['unseen_test_rows']} test rows with levels {s['unseen_test_levels']}")
        if s["drift_tvd"] >= DRIFT_TVD:
            flag("train_test_drift", c, f"total-variation distance {s['drift_tvd']}")
        if s["n_levels"] > 0.5 * len(train):
            flag("id_like", c, f"{s['n_levels']} unique values in {len(train)} rows")
    for c, r in prof["linear_redundancy"].items():
        if "formula" in r:
            flag("exact_linear_dependency", c, f"{c} = {r['formula']} + {r['formula_intercept']} (R^2={r['formula_r2']})")
    if prof["high_corr_pairs"]:
        top = sorted(prof["high_corr_pairs"], key=lambda q: -abs(q["r"]))
        cols = sorted({q["a"] for q in top} | {q["b"] for q in top})
        flag("high_correlation", cols, f"{len(top)} pairs with |r| >= {HIGH_CORR}; strongest: " +
             ", ".join(f"{q['a']}~{q['b']} r={q['r']}" for q in top[:4]))
    for c, auc in prof["single_feature_auc"].items():
        if auc >= LEAK_SCORE:
            flag("possible_target_leakage", c, f"single-feature CV AUC {auc}")
    if prof["duplicates"]["full_rows"]:
        flag("duplicate_rows", None, f"{prof['duplicates']['full_rows']} exact duplicate train rows")
    if prof["duplicates"]["test_rows_also_in_train"]:
        flag("train_test_overlap", None, f"{prof['duplicates']['test_rows_also_in_train']} test feature rows also in train")
    prof["flags"] = flags
    prof["n_flags"] = len(flags)

    dump_json(prof, out / "profile.json")
    (out / "profile.md").write_text(render_digest(prof))
    print(f"profile written: {out/'profile.json'} ({len(flags)} flags)")


def render_digest(p: dict) -> str:
    L = [f"# Data profile digest\n",
         f"- task: **{p['task']}**, target `{p['target']}`; train {p['n_train']} rows, test {p['n_test']} rows "
         f"({p['load']['test_source']}); {p['n_features']} features "
         f"({len(p['numeric_columns'])} numeric, {len(p['categorical_columns'])} categorical); "
         f"one-hot width ~{p['onehot_width_estimate']}",
         f"- target (train): {p['target_train']}; minority share {p['minority_share']}",
         f"- rows with any missing feature: train {p['missing']['rows_with_any_missing_train']}, "
         f"test {p['missing']['rows_with_any_missing_test']}",
         f"- duplicates: {p['duplicates']}", "", "## Numeric columns",
         "| col | miss | min | q1 | med | q3 | max | skew | zeros | >3IQR | KS |", "|" + "---|" * 11]
    for c, s in p["numeric"].items():
        L.append(f"| {c} | {s['missing_train']}/{s['missing_test']} | {s['min']} | {s['q1']} | {s['median']} | "
                 f"{s['q3']} | {s['max']} | {s['skew']} | {s['zeros']} | {s['outlier_share_3iqr']} | {s['drift_ks']} |")
    L += ["", "## Categorical columns", "| col | miss | levels | dominant | rare(<1%) | placeholders | +rate by level |",
          "|" + "---|" * 7]
    for c, s in p["categorical"].items():
        L.append(f"| {c} | {s['missing_train']}/{s['missing_test']} | {s['n_levels']} | {s['dominant_share']} | "
                 f"{s['rare_levels']} ({s['rare_share_total']}) | {s['placeholder_levels']} | "
                 f"{s.get('positive_rate_by_level', '')} |")
    L += ["", "## Single-feature predictive power (CV AUC of depth-4 tree)", str(p["single_feature_auc"]),
          "", "## Linear redundancy (R^2 from other numerics)",
          str({c: r for c, r in p["linear_redundancy"].items()}), "", "## Flags"]
    L += [f"- **{f['kind']}** `{f['column']}`: {f['evidence']}" for f in p["flags"]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
