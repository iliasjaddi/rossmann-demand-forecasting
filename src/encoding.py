"""Hold one implementation of the categorical encoding.

Every path uses these functions: training, serving-data build, and the API.

LightGBM maps a categorical feature by code position, not by value. The
serving path must therefore build its categories in the same order as the
training path. If the orders differ, the model reads each category as a
different one.

This module prevents a specific fault. An earlier version rebuilt the
categories at serving time and cast them to string first. Strings sort as
'1', '10', '100'. Integers sort as 1, 2, 3. The model then scored store 262
as a different store, and returned a wrong value with no error message.

Parquet does not preserve the pandas category dtype. The encoding therefore
cannot be stored in the file. Each path calls the same two functions here.
check_serving_skew.py tests that the paths agree.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CATEGORICAL = ["Store", "StoreType", "Assortment", "StateHoliday", "DayOfWeek"]


def training_categories(features_parquet=None):
    """Return the category lists that training used.

    This repeats what features.py does: `astype("category")` on the raw dtype.
    pandas sorts the values it finds. Integer columns sort numerically. String
    columns sort alphabetically. The function does not change the dtype before
    the cast, so both orders stay correct.
    """
    path = features_parquet or (ROOT / "features.parquet")
    df = pd.read_parquet(path, columns=CATEGORICAL)
    return {c: df[c].astype("category").cat.categories.tolist()
            for c in CATEGORICAL}


def apply_categories(df, cats, strict=True):
    """Apply the stored categories to a frame and keep each column dtype.

    With `strict`, the function raises an error if a value is outside the
    training categories. Without it, that value becomes NaN and the model
    reads it as a missing value.
    """
    df = df.copy()
    for col, values in cats.items():
        if col not in df.columns:
            continue
        target = pd.Index(values)
        col_cast = df[col].astype(target.dtype)
        df[col] = pd.Categorical(col_cast, categories=target)
        if strict:
            unmapped = int(df[col].isna().sum() - col_cast.isna().sum())
            if unmapped:
                raise ValueError(
                    f"{unmapped} values of '{col}' are outside the training "
                    f"categories; the encoding would not match the model."
                )
    return df


def save_categories(cats, path):
    Path(path).write_text(json.dumps(cats, indent=2))


def load_categories(path):
    return json.loads(Path(path).read_text())
