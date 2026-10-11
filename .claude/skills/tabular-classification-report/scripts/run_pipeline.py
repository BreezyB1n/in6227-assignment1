"""Stage 3 - train, tune and evaluate exactly what plan.json says.

All preprocessing lives inside an sklearn Pipeline, so every statistic
(medians, quantiles, encoders, scalers) is re-fitted inside each CV fold and
the test set is used once, after all choices are frozen.

Usage:
    python run_pipeline.py --data ... [--test ...] [--target ...] --plan runs/x/plan.json --out runs/x
Writes runs/x/results.json and runs/x/fig_results.pdf (+ .png).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import column_types, dump_json, load_dataset, split_xy

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ESTIMATORS = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "GaussianNB": GaussianNB,
}
BINARY_SCORING = ["average_precision", "roc_auc", "f1", "f1_macro", "balanced_accuracy",
                  "accuracy", "precision", "recall"]
MULTI_SCORING = ["f1_macro", "balanced_accuracy", "accuracy", "roc_auc_ovr", "neg_log_loss"]


# ----------------------------------------------------------------- transformers
class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Treat exact zeros in the given columns as missing (stateless)."""
    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            if c in X:
                X.loc[X[c] == 0, c] = np.nan
        return X


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise numeric columns at quantiles learned on the training fold."""
    def __init__(self, q=(0.005, 0.995)):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = np.nanquantile(X, self.q[0], axis=0), np.nanquantile(X, self.q[1], axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class Log1p(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(np.clip(np.asarray(X, dtype=float), 0, None))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


# ----------------------------------------------------------------- pipeline build
def _decision(plan, key, default):
    v = plan.get(key)
    if v is None:
        return default
    return v.get("value", v.get("columns", default)) if isinstance(v, dict) else v


def build_pipeline(plan: dict, model: dict, num: list[str], cat: list[str], task: str, seed: int):
    # model-level "preprocess" entries override the dataset-level decisions of the plan
    plan = {**plan, **model.get("preprocess", {})}
    drop = set(_decision(plan, "drop_columns", []))
    num = [c for c in num if c not in drop]
    cat = [c for c in cat if c not in drop]
    zero_cols = [c for c in _decision(plan, "zero_as_missing", []) if c in num]
    clip_cfg = plan.get("clip_quantiles") or {}
    clip_cols = [c for c in clip_cfg.get("columns", []) if c in num]
    log_cols = [c for c in _decision(plan, "log_transform", []) if c in num]
    indicators = bool(_decision(plan, "missing_indicators", True))
    rare = float(_decision(plan, "rare_level_threshold", 0.0)) or None
    cat_missing = _decision(plan, "categorical_missing", "own_level")
    encoding = model.get("encoding", "onehot")
    scale = bool(model.get("scale", encoding == "onehot"))

    # numeric branch: optional clip / log on subsets, then impute (+ indicator) and scale
    steps = []
    special = []
    if clip_cols:
        special.append(("clip", Pipeline([("clip", QuantileClipper(tuple(clip_cfg.get("q", (0.005, 0.995)))))]),
                        [c for c in clip_cols if c not in log_cols]))
    if log_cols:
        special.append(("log", Pipeline([("log", Log1p())]), log_cols))
    plain = [c for c in num if c not in clip_cols and c not in log_cols]
    if special:
        steps.append(("shape", ColumnTransformer(special + [("keep", "passthrough", plain)],
                                                  verbose_feature_names_out=False)))
    steps.append(("impute", SimpleImputer(strategy="median", add_indicator=indicators)))
    if scale:
        steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(steps)

    cat_impute = (SimpleImputer(strategy="constant", fill_value="__missing__") if cat_missing == "own_level"
                  else SimpleImputer(strategy="most_frequent"))
    if encoding == "onehot":
        enc = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=rare, sparse_output=False)
    else:  # "ordinal" or "native" (native = HistGradientBoosting categorical support)
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=rare)
    cat_pipe = Pipeline([("impute", cat_impute), ("encode", enc)])

    # categorical block first: its width is fixed, whereas the numeric block grows with
    # missing-indicator columns that depend on the training fold
    pre = ColumnTransformer([("cat", cat_pipe, cat), ("num", num_pipe, num)], verbose_feature_names_out=False)
    params = dict(model.get("params", {}))
    est_cls = ESTIMATORS[model["estimator"]]
    if "random_state" in est_cls().get_params():
        params.setdefault("random_state", seed)
    est = est_cls(**params)
    pipe = Pipeline([("zero", ZeroToNaN(zero_cols)), ("pre", pre), ("clf", est)])
    return pipe, {"numeric": num, "categorical": cat, "dropped": sorted(drop), "zero_as_missing": zero_cols,
                  "clipped": clip_cols, "logged": log_cols, "encoding": encoding, "scaled": scale}


def finalise_native(pipe: Pipeline, n_cat: int):
    """Native categorical support for HistGradientBoosting: the first `n_cat` columns are categorical."""
    if not n_cat or not isinstance(pipe.named_steps["clf"], HistGradientBoostingClassifier):
        return pipe
    enc = pipe.named_steps["pre"].transformers[0][1].named_steps["encode"]
    # HGB needs non-negative codes; categories unseen in training become NaN (HGB's missing branch)
    enc.set_params(unknown_value=np.nan)
    pipe.named_steps["clf"].set_params(categorical_features=list(range(n_cat)))
    return pipe


# ----------------------------------------------------------------- search space
def to_distribution(spec):
    if isinstance(spec, list):
        return spec
    (kind, (lo, hi)), = spec.items()
    if kind == "loguniform":
        return stats.loguniform(lo, hi)
    if kind == "uniform":
        return stats.uniform(lo, hi - lo)
    if kind == "randint":
        return stats.randint(lo, hi + 1)
    raise ValueError(f"unknown distribution {spec}")


def at_boundary(spec, value):
    if isinstance(spec, list) or value is None:
        return False
    (kind, (lo, hi)), = spec.items()
    # wide positive ranges (hi/lo >= 20) are judged on a log scale, otherwise a value
    # such as 30 in [10, 600] would count as "at the boundary"
    if kind == "loguniform" or (lo > 0 and hi / lo >= 20):
        span = np.log(hi) - np.log(lo)
        return min(np.log(value) - np.log(lo), np.log(hi) - np.log(value)) < 0.05 * span
    return min(value - lo, hi - value) < 0.05 * (hi - lo)


# ----------------------------------------------------------------- metrics
def binary_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {"threshold": thr, "accuracy": accuracy_score(y, yhat), "balanced_accuracy": balanced_accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat),
            "f1": f1_score(y, yhat), "f1_macro": f1_score(y, yhat, average="macro"),
            "confusion": confusion_matrix(y, yhat).tolist()}


def multi_metrics(y, P):
    yhat = P.argmax(1)
    return {"accuracy": accuracy_score(y, yhat), "balanced_accuracy": balanced_accuracy_score(y, yhat),
            "f1_macro": f1_score(y, yhat, average="macro"),
            "f1_per_class": f1_score(y, yhat, average=None).tolist(),
            "roc_auc_ovr": roc_auc_score(y, P, multi_class="ovr", average="macro"),
            "log_loss": log_loss(y, P, labels=list(range(P.shape[1]))),
            "confusion": confusion_matrix(y, yhat).tolist()}


def objective_at(y, p, t, objective):
    yhat = (p >= t).astype(int)
    if objective == "f1":
        return float(f1_score(y, yhat))
    if objective == "f1_macro":
        return float(f1_score(y, yhat, average="macro"))
    return float(balanced_accuracy_score(y, yhat))


def best_threshold(y, p, objective):
    grid = np.round(np.linspace(0.05, 0.95, 91), 2)
    scores = np.array([objective_at(y, p, t, objective) for t in grid])
    return float(grid[scores.argmax()]), float(scores.max())


def bootstrap(y, preds: dict, metric_fns: dict, B: int, seed: int):
    """Percentile CIs per model, plus paired CI of (model B - model A) for the first two models."""
    rng = np.random.default_rng(seed)
    n = len(y)
    names = list(preds)
    draws = {m: {k: [] for k in metric_fns} for m in names}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        for m in names:
            for k, fn in metric_fns.items():
                draws[m][k].append(fn(y[idx], preds[m], idx))
    ci = {m: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()}
          for m, d in draws.items()}
    diff = {}
    if len(names) >= 2:
        a, b = names[0], names[1]
        for k in metric_fns:
            d = np.array(draws[b][k]) - np.array(draws[a][k])
            diff[k] = {"mean": float(d.mean()), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                       "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    return ci, diff


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio corrected resampled t-test on paired CV fold scores (b - a)."""
    d = np.asarray(b) - np.asarray(a)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return {"mean_diff": float(d.mean()), "t": None, "p": None}
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return {"mean_diff": float(d.mean()), "t": float(t), "p": float(2 * stats.t.sf(abs(t), k - 1))}


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--test")
    ap.add_argument("--target")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    plan = json.loads(pathlib.Path(a.plan).read_text())
    seed = plan.get("seed", a.seed)
    ds = load_dataset(a.data, a.test, a.target, out, seed=seed)
    target = ds["target"]
    X, y_raw = split_xy(ds["train"], target)
    Xt, yt_raw = split_xy(ds["test"], target)
    num, cat = column_types(X)
    classes = sorted(y_raw.unique().tolist())
    task = "binary" if len(classes) == 2 else "multiclass"
    if task == "binary":
        pos = plan.get("positive_class", y_raw.value_counts().idxmin())
        pos = type(classes[0])(pos) if not isinstance(pos, type(classes[0])) else pos
        y, yt = (y_raw == pos).astype(int).to_numpy(), (yt_raw == pos).astype(int).to_numpy()
        class_names = [c for c in classes if c != pos] + [pos]
        scoring = BINARY_SCORING
    else:
        pos = None
        index = {c: i for i, c in enumerate(classes)}
        y, yt = y_raw.map(index).to_numpy(), yt_raw.map(index).to_numpy()
        class_names = classes
        scoring = MULTI_SCORING
    cv_cfg = plan.get("cv", {})
    folds = int(cv_cfg.get("folds", 5))
    primary = cv_cfg.get("primary_metric", "average_precision" if task == "binary" else "f1_macro")
    if primary not in scoring:
        raise ValueError(f"primary metric {primary} not in {scoring}")
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    thr_cfg = plan.get("threshold", {})
    thr_objective = thr_cfg.get("objective", "f1") if task == "binary" else None
    tune_thr = task == "binary" and thr_cfg.get("strategy", "oof") != "none"

    res = {"task": task, "target": target, "positive_class": pos, "class_names": [str(c) for c in class_names],
           "load": ds["log"], "n_train": len(X), "n_test": len(Xt), "test_prevalence": float(yt.mean()) if task == "binary" else None,
           "cv": {"folds": folds, "primary_metric": primary, "seed": seed}, "models": {}}

    # ---- reference: majority / prior baseline
    dummy = DummyClassifier(strategy="prior").fit(X, y)
    pd_ = dummy.predict_proba(Xt)
    if task == "binary":
        res["baseline"] = {"test": {"average_precision": average_precision_score(yt, pd_[:, 1]),
                                    "roc_auc": 0.5, **binary_metrics(yt, pd_[:, 1], 0.5)}}
    else:
        res["baseline"] = {"test": multi_metrics(yt, pd_)}

    test_proba, test_pred, fold_scores = {}, {}, {}
    fitted = {}
    for m in plan["models"]:
        mid = m["id"]
        t0 = time.time()
        pipe, prep = build_pipeline(plan, m, num, cat, task, seed)
        if m.get("encoding") == "native":
            pipe = finalise_native(pipe, len(prep["categorical"]))
        space = {f"clf__{k}": to_distribution(v) for k, v in m.get("search", {}).items()}
        n_iter = int(m.get("n_iter", 20))
        if space:
            n_grid = np.prod([len(v) for v in space.values() if isinstance(v, list)]) if all(
                isinstance(v, list) for v in space.values()) else None
            search = RandomizedSearchCV(pipe, space, n_iter=min(n_iter, n_grid) if n_grid else n_iter,
                                        scoring=scoring, refit=primary, cv=cv, n_jobs=-1,
                                        random_state=seed, return_train_score=True, error_score="raise")
            search.fit(X, y)
            best = search.best_estimator_
            cvr, bi = search.cv_results_, search.best_index_
            best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
            n_tried = len(cvr["params"])
        else:
            cvr = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
            cvr = {f"{k.replace('test_', 'split_test_')}": v for k, v in cvr.items()}
            best = clone(pipe).fit(X, y)
            bi, best_params, n_tried = None, {}, 1
        def cvstat(metric, split="test"):
            if bi is None:
                v = np.asarray(cvr[f"split_{split}_{metric}"]) if split == "test" else np.asarray(cvr[f"train_{metric}"])
                return float(v.mean()), float(v.std()), v.tolist()
            ks = [f"split{i}_{split}_{metric}" for i in range(folds)]
            v = np.array([cvr[k][bi] for k in ks])
            return float(v.mean()), float(v.std()), v.tolist()
        cv_summary = {k: {"mean": cvstat(k)[0], "std": cvstat(k)[1]} for k in scoring}
        train_primary = cvstat(primary, "train")[0]
        fold_scores[mid] = cvstat(primary)[2]
        search_time = time.time() - t0

        clf = best.named_steps["clf"]
        fit_info = {}
        if hasattr(clf, "n_iter_"):
            it = np.atleast_1d(clf.n_iter_).max()
            fit_info["n_iter"] = int(it)
            mx = clf.get_params().get("max_iter")
            if mx:
                fit_info["max_iter"] = int(mx)
                fit_info["stopped_early_or_converged"] = bool(it < mx)
        if hasattr(clf, "get_depth"):
            fit_info["tree_depth"] = int(clf.get_depth())
        boundary = [k for k, v in m.get("search", {}).items() if at_boundary(v, best_params.get(k))]
        n_features_in = len(best.named_steps["pre"].get_feature_names_out())

        # ---- out-of-fold probabilities -> decision threshold (never uses test)
        if task == "binary":
            oof = cross_val_predict(clone(best), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
            thr, thr_score = best_threshold(y, oof, thr_objective) if tune_thr else (0.5, None)
            p = best.predict_proba(Xt)[:, 1]
            test = {"average_precision": average_precision_score(yt, p), "roc_auc": roc_auc_score(yt, p),
                    "at_0.5": binary_metrics(yt, p, 0.5), "at_tuned": binary_metrics(yt, p, thr)}
            oof_info = {"threshold": thr, "objective": thr_objective, "oof_objective_at_threshold": thr_score,
                        "oof_objective_at_0.5": objective_at(y, oof, 0.5, thr_objective),
                        "oof_average_precision": average_precision_score(y, oof)}
            test_proba[mid] = p
            test_pred[mid] = (p >= thr).astype(int)
        else:
            P = best.predict_proba(Xt)
            test = multi_metrics(yt, P)
            oof_info = {}
            test_proba[mid] = P
            test_pred[mid] = P.argmax(1)
        fitted[mid] = best
        res["models"][mid] = {"estimator": m["estimator"], "label": m.get("label", m["estimator"]),
                              "preprocessing": prep, "n_features_after_preprocessing": n_features_in,
                              "fixed_params": m.get("params", {}), "search_space": m.get("search", {}),
                              "n_candidates": n_tried, "best_params": best_params, "params_at_search_boundary": boundary,
                              "cv": cv_summary, "cv_train_primary": train_primary,
                              "overfit_gap_primary": train_primary - cv_summary[primary]["mean"],
                              "fit": fit_info, "search_seconds": round(search_time, 1),
                              "oof": oof_info, "test": test}
        print(f"[{mid}] cv {primary}={cv_summary[primary]['mean']:.4f}±{cv_summary[primary]['std']:.4f} "
              f"best={best_params} ({search_time:.0f}s)")

    ids = [m["id"] for m in plan["models"]]
    # ---- statistical comparison of the first two models
    if len(ids) >= 2:
        A, B = ids[0], ids[1]
        n_tr = len(X) * (folds - 1) / folds
        comp = {"pair": [A, B], "cv_corrected_ttest": corrected_ttest(fold_scores[A], fold_scores[B], n_tr, len(X) / folds)}
        ra, rb = test_pred[A] == yt, test_pred[B] == yt
        b_, c_ = int((ra & ~rb).sum()), int((~ra & rb).sum())
        comp["mcnemar"] = {"A_right_B_wrong": b_, "A_wrong_B_right": c_,
                           "p_exact": float(stats.binomtest(min(b_, c_), b_ + c_, 0.5).pvalue) if b_ + c_ else 1.0}
        if task == "binary":
            thr = {k: res["models"][k]["oof"]["threshold"] for k in ids}
            fns = {"average_precision": lambda yy, p, i: average_precision_score(yy, p[i]),
                   "roc_auc": lambda yy, p, i: roc_auc_score(yy, p[i])}
            ci_a, diff = bootstrap(yt, {A: test_proba[A], B: test_proba[B]}, fns, a.bootstrap, seed)
            fns2 = {"f1": lambda yy, p, i: f1_score(yy, p[i]), "f1_macro": lambda yy, p, i: f1_score(yy, p[i], average="macro"),
                    "balanced_accuracy": lambda yy, p, i: balanced_accuracy_score(yy, p[i])}
            ci_b, diff2 = bootstrap(yt, {A: test_pred[A], B: test_pred[B]}, fns2, a.bootstrap, seed)
            comp["bootstrap_ci"] = {k: {**ci_a[k], **ci_b[k]} for k in (A, B)}
            comp["bootstrap_diff_B_minus_A"] = {**diff, **diff2}
        else:
            fns = {"f1_macro": lambda yy, p, i: f1_score(yy, p[i], average="macro"),
                   "balanced_accuracy": lambda yy, p, i: balanced_accuracy_score(yy, p[i])}
            ci, diff = bootstrap(yt, {A: test_pred[A], B: test_pred[B]}, fns, a.bootstrap, seed)
            comp["bootstrap_ci"], comp["bootstrap_diff_B_minus_A"] = ci, diff
        res["comparison"] = comp

    # ---- permutation importance on the test set (model inspection only, no selection)
    scorer = primary if primary != "neg_log_loss" else "f1_macro"
    for mid, est in fitted.items():
        pi = permutation_importance(est, Xt, yt, scoring=scorer, n_repeats=5, random_state=seed, n_jobs=-1)
        order = np.argsort(-pi.importances_mean)
        dropped = set(res["models"][mid]["preprocessing"]["dropped"])
        res["models"][mid]["permutation_importance"] = {
            "scoring": scorer,
            "top": [{"feature": Xt.columns[i], "mean": float(pi.importances_mean[i]), "std": float(pi.importances_std[i])}
                    for i in order[:8]],
            "near_zero": [Xt.columns[i] for i in order
                          if abs(pi.importances_mean[i]) < 1e-3 and Xt.columns[i] not in dropped]}

    # ---- ablations: change one plan decision, re-run CV with the tuned hyper-parameters
    res["ablations"] = []
    for ab in plan.get("ablations", []):
        base_model = next(m for m in plan["models"] if m["id"] == ab["model"])
        variant_plan = plan
        variant_model = {**base_model, **ab.get("model_change", {}),
                         "preprocess": {**base_model.get("preprocess", {}), **ab.get("change", {})},
                         "params": {**base_model.get("params", {}), **res["models"][ab["model"]]["best_params"],
                                    **ab.get("model_change", {}).get("params", {})}}
        pipe, vprep = build_pipeline(variant_plan, variant_model, num, cat, task, seed)
        if variant_model.get("encoding") == "native":
            pipe = finalise_native(pipe, len(vprep["categorical"]))
        scores = cross_validate(pipe, X, y, cv=cv, scoring=primary, n_jobs=-1)["test_score"]
        ref = fold_scores[ab["model"]]
        res["ablations"].append({"id": ab["id"], "model": ab["model"], "question": ab.get("question", ""),
                                 "cv_mean": float(scores.mean()), "cv_std": float(scores.std()),
                                 "reference_cv_mean": float(np.mean(ref)),
                                 "delta": float(scores.mean() - np.mean(ref)),
                                 # sd of the paired per-fold differences: |delta| below it = no detectable effect
                                 "delta_fold_sd": float(np.std(np.asarray(scores) - np.asarray(ref), ddof=1)),
                                 "within_noise": bool(abs(scores.mean() - np.mean(ref)) <
                                                      np.std(np.asarray(scores) - np.asarray(ref), ddof=1)),
                                 "corrected_ttest": corrected_ttest(ref, scores, len(X) * (folds - 1) / folds, len(X) / folds)})
        print(f"[ablation {ab['id']}] {primary} {scores.mean():.4f} (ref {np.mean(ref):.4f})")

    make_figure(res, ids, yt, test_proba, test_pred, class_names, out)
    dump_json(res, out / "results.json")
    print(f"results written: {out/'results.json'}")


def make_figure(res, ids, yt, proba, pred, class_names, out):
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "font.family": "serif"})
    colors = ["#1f5f99", "#c2571a"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.85), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    if res["task"] == "binary":
        for k, col in zip(ids, colors):
            pr, rc, _ = precision_recall_curve(yt, proba[k])
            m = res["models"][k]
            ax.plot(rc, pr, color=col, lw=1.2, label=f"{k.upper()} (AP={m['test']['average_precision']:.3f})")
            op = m["test"]["at_tuned"]
            ax.plot(op["recall"], op["precision"], "o", color=col, ms=4, mec="black", mew=0.5)
        ax.axhline(res["test_prevalence"], color="grey", ls="--", lw=0.8, label=f"prior ({res['test_prevalence']:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("(a) Test precision-recall")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.25), frameon=False)
    else:
        x = np.arange(len(class_names)); w = 0.38
        for j, (k, col) in enumerate(zip(ids, colors)):
            ax.bar(x + (j - 0.5) * w, res["models"][k]["test"]["f1_per_class"], w, color=col, label=k.upper())
        ax.set_xticks(x); ax.set_xticklabels([str(c)[:8] for c in class_names], rotation=30)
        ax.set_ylabel("F1"); ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title("(a) Test F1 per class"); ax.legend(loc="upper center", ncol=2, frameon=False)
    for j, (k, ax) in enumerate(zip(ids[:2], axes[1:])):
        cm = np.array(res["models"][k]["test"]["at_tuned"]["confusion"] if res["task"] == "binary"
                      else res["models"][k]["test"]["confusion"])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for (i, jj), v in np.ndenumerate(cm):
            ax.text(jj, i, f"{v}\n{norm[i, jj]:.0%}", ha="center", va="center", fontsize=6,
                    color="white" if norm[i, jj] > 0.6 else "black")
        labels = [str(c)[:8] for c in class_names]
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30 if len(labels) > 2 else 0)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        thr = f", t={res['models'][k]['test']['at_tuned']['threshold']:.2f}" if res["task"] == "binary" else ""
        ax.set_title(f"({'bc'[j]}) {res['models'][k]['label']}{thr}")
    fig.tight_layout(pad=0.4, w_pad=1.0)
    fig.savefig(out / "fig_results.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    main()
