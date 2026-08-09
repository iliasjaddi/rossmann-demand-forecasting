"""Test the training path and the serving path for differences.

The statement to test: the feature table that the API uses gives the same
predictions as the offline pipeline, for each row that both contain.

This test found a real fault. The serving table built its categorical
encodings again from the start and cast the values to string first. Strings
sort as '1', '10', '100', '1000'. Integers sort as 1, 2, 3, 4. LightGBM maps
a categorical feature by code position. Each store therefore became a
different store. The API returned 8,067 where the offline pipeline returned
18,822. The actual value was 19,894.

The model was correct. No code raised an error. The response had the correct
format. Only a comparison of the two paths on the same rows can show the
fault.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from encoding import CATEGORICAL, apply_categories, load_categories

ROOT = Path(__file__).resolve().parent.parent
SERVE = ROOT / "serving"
TOL = 1e-6


def main():
    meta = json.loads((SERVE / "model_meta.json").read_text())
    feats = meta["features"]
    model = lgb.Booster(model_file=str(SERVE / "model.txt"))

    # Offline path: cast exactly as features.py did at training time.
    offline = pd.read_parquet(ROOT / "features.parquet")
    offline["store_key"] = offline["Store"].astype(int)
    for c in CATEGORICAL:
        offline[c] = offline[c].astype("category")

    # Serving path: the frozen categories, applied through the shared helper,
    # which is the same code the API runs.
    serving = apply_categories(
        pd.read_parquet(SERVE / "serving_features.parquet"),
        load_categories(SERVE / "categories.json"),
    )

    # Rows both paths hold: our sealed test window.
    key = ["store_key", "Date"]
    common = offline.merge(serving[key], on=key, how="inner")
    common = common.sort_values(key).reset_index(drop=True)
    srv = serving.merge(common[key], on=key, how="inner")
    srv = srv.sort_values(key).reset_index(drop=True)

    print(f"rows compared: {len(common):,}")
    assert len(common) == len(srv) and (
        common.store_key.to_numpy() == srv.store_key.to_numpy()).all()

    p_off = np.expm1(model.predict(common[feats]))
    p_srv = np.expm1(model.predict(srv[feats]))

    diff = np.abs(p_off - p_srv)
    n_bad = int((diff > TOL).sum())

    print(f"max absolute difference: {diff.max():.10f}")
    if n_bad == 0:
        print("\nPASS: serving path and offline path agree exactly.")
    else:
        worst = np.argsort(-diff)[:5]
        print(f"\nFAIL: {n_bad:,} rows disagree ({n_bad / len(diff) * 100:.2f}%)\n")
        print(pd.DataFrame({
            "store": common.store_key.to_numpy()[worst],
            "date": common.Date.to_numpy()[worst],
            "offline": p_off[worst].round(1),
            "serving": p_srv[worst].round(1),
        }).to_string(index=False))
        raise SystemExit(1)

    # Second guard: the categorical dtypes themselves must match, not just
    # the predictions on this particular sample.
    print("\ncategory alignment:")
    for c in CATEGORICAL:
        if c not in feats:
            continue
        a = list(offline[c].cat.categories)
        b = list(serving[c].cat.categories)
        status = "ok" if a == b else "MISMATCH"
        print(f"  {c:14s} {len(a):>5} categories  {status}")
        if a != b:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
