"""Shared data loading for the tabular-classification-report scripts.

Every script loads data through `load_dataset` so that profiling, training and
reporting all see exactly the same rows, the same target and the same split.
"""
from __future__ import annotations

import json
import pathlib
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_HINTS = ("label", "target", "class", "y", "outcome", "income", "churn")
EXTRA_NA = ["?", "", " ", "NA", "N/A", "n/a", "null", "NULL", "None", "nan", "NaN"]
TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".parquet", ".xlsx", ".xls")


def read_table(path: pathlib.Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path, na_values=EXTRA_NA, skipinitialspace=True)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", na_values=EXTRA_NA, skipinitialspace=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, na_values=EXTRA_NA)
    raise ValueError(f"Unsupported table format: {path}")


def resolve_inputs(data: str, test: str | None, workdir: pathlib.Path):
    """Return (train_path, test_path | None). Accepts a file, a directory or a .zip."""
    p = pathlib.Path(data).expanduser()
    if p.suffix.lower() == ".zip":
        out = workdir / "_data"
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p) as z:
            members = [m for m in z.namelist()
                       if not m.startswith("__MACOSX") and not pathlib.Path(m).name.startswith(".")]
            z.extractall(out, members)  # zipfile strips absolute paths and '..'
        p = out
    test_path = pathlib.Path(test).expanduser() if test else None
    if p.is_dir():
        files = sorted(f for f in p.rglob("*")
                       if f.suffix.lower() in TABLE_SUFFIXES and not f.name.startswith("."))
        train = [f for f in files if "train" in f.name.lower()]
        tests = [f for f in files if "test" in f.name.lower()]
        if train:
            return train[0], test_path or (tests[0] if tests else None)
        if len(files) == 1:
            return files[0], test_path
        raise ValueError(f"Cannot tell train/test apart in {p}: {[f.name for f in files]}")
    return p, test_path


def infer_target(df: pd.DataFrame, target: str | None) -> str:
    if target:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not in {list(df.columns)}")
        return target
    lower = {c.lower(): c for c in df.columns}
    for hint in TARGET_HINTS:
        if hint in lower:
            return lower[hint]
    return df.columns[-1]


def _normalise_strings(df: pd.DataFrame) -> int:
    changed = 0
    for c in df.select_dtypes(include="object").columns:
        stripped = df[c].str.strip()
        changed += int((stripped != df[c]).fillna(False).sum())
        df[c] = stripped
    return changed


def load_dataset(data: str, test: str | None, target: str | None,
                 workdir: pathlib.Path, seed: int = 42, test_size: float = 0.2) -> dict:
    """Load, lightly sanitise and split. Returns a dict with train/test frames and a log.

    Only label-level hygiene happens here (it is never a modelling choice):
    whitespace stripping, label spelling reconciliation and dropping rows whose
    label is missing. All feature-level decisions belong to the plan.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = resolve_inputs(data, test, workdir)
    train = read_table(train_path)
    target = infer_target(train, target)
    # file names only: absolute paths would leak local directory names into published artefacts
    log = {"train_file": train_path.name, "test_file": test_path.name if test_path else None,
           "target": target, "raw_train_rows": len(train)}
    log["whitespace_cells_stripped"] = _normalise_strings(train)

    if test_path is not None:
        test_df = read_table(test_path)
        log["raw_test_rows"] = len(test_df)
        log["whitespace_cells_stripped"] += _normalise_strings(test_df)
        if target not in test_df.columns:
            raise ValueError(f"Test file has no target column '{target}'")
        # Reconcile label spellings such as '>50K.' vs '>50K' (classic UCI Adult issue).
        train_labels = set(train[target].dropna().astype(str))
        fixes = {}
        for v in set(test_df[target].dropna().astype(str)) - train_labels:
            if v.rstrip(".") in train_labels:
                fixes[v] = v.rstrip(".")
        if fixes:
            test_df[target] = test_df[target].astype(str).replace(fixes)
        log["label_spelling_fixes"] = fixes
        log["test_source"] = "provided"
    else:
        test_df = None
        log["test_source"] = f"stratified {int(test_size * 100)}% hold-out (seed {seed})"

    log["train_rows_missing_label"] = int(train[target].isna().sum())
    train = train[train[target].notna()].reset_index(drop=True)
    if test_df is not None:
        log["test_rows_missing_label"] = int(test_df[target].isna().sum())
        test_df = test_df[test_df[target].notna()].reset_index(drop=True)
    else:
        log["test_rows_missing_label"] = 0
        train, test_df = train_test_split(train, test_size=test_size, random_state=seed,
                                          stratify=train[target])
        train, test_df = train.reset_index(drop=True), test_df.reset_index(drop=True)

    # Integer-coded binary labels stay as-is; everything else becomes string.
    for df in (train, test_df):
        if df[target].dtype == object or str(df[target].dtype).startswith("category"):
            df[target] = df[target].astype(str)
    log["train_rows"], log["test_rows"] = len(train), len(test_df)
    return {"train": train, "test": test_df, "target": target, "log": log}


def split_xy(df: pd.DataFrame, target: str):
    return df.drop(columns=[target]), df[target]


def column_types(X: pd.DataFrame):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return num, cat


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, pathlib.Path):
            return str(o)
        return super().default(o)


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    return obj


def dump_json(obj, path: pathlib.Path):
    path.write_text(json.dumps(_clean(obj), indent=2, cls=NumpyEncoder))
