"""Stage 4: gradient boosted model.

Trained on log1p(Sales) over trading days only, with early stopping on the
validation window. The test window is not touched here.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import DROP, build, feature_names
from metrics import evaluate, results_table

ROOT = Path(__file__).resolve().parent.parent
FEATURE_CACHE = ROOT / "features.parquet"

PARAMS = {
    "objective": "regression",       # L2 loss, applied to the LOG target,
    "metric": "rmse",                # which makes it ~relative error in euros
    "learning_rate": 0.05,
    "num_leaves": 128,               # leaf-wise growth: capacity knob
    "min_data_in_leaf": 50,          # main brake on overfitting
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_cat_threshold": 64,         # Store has 1,115 levels
    "num_threads": 0,
    "verbosity": -1,
    "seed": 42,
}
NUM_ROUNDS = 4000
EARLY_STOPPING = 150


def load_features(rebuild=False):
    if FEATURE_CACHE.exists() and not rebuild:
        df = pd.read_parquet(FEATURE_CACHE)
        for c in ["Store", "StoreType", "Assortment", "StateHoliday", "DayOfWeek"]:
            df[c] = df[c].astype("category")
        return df
    return build()


def make_datasets(df, feats, drop_store=False):
    """Trading days only. Closed days are a deterministic zero, not a forecast."""
    use = [f for f in feats if not (drop_store and f == "Store")]
    tr = df[(df.split == "train") & (df.Open == 1)]
    va = df[(df.split == "valid") & (df.Open == 1)]

    dtrain = lgb.Dataset(tr[use], label=np.log1p(tr.Sales))
    dvalid = lgb.Dataset(va[use], label=np.log1p(va.Sales), reference=dtrain)
    return dtrain, dvalid, use, tr, va


def predict_full(model, df, use, split):
    """Predict a whole split, then overwrite closed days with exact zero."""
    sub = df[df.split == split]
    pred = np.expm1(model.predict(sub[use], num_iteration=model.best_iteration))
    pred = np.clip(pred, 0, None)
    pred[sub.Open.to_numpy() == 0] = 0.0
    return sub, pred


def run(drop_store=False, params=None, rebuild=False):
    df = load_features(rebuild)
    feats = feature_names(df)
    params = {**PARAMS, **(params or {})}

    dtrain, dvalid, use, tr, va = make_datasets(df, feats, drop_store)
    print(f"train rows: {len(tr):,}   valid rows: {len(va):,}   "
          f"features: {len(use)}\n")

    model = lgb.train(
        params, dtrain,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(250),
        ],
    )
    print(f"\nbest iteration: {model.best_iteration} of {NUM_ROUNDS}")

    sub, pred = predict_full(model, df, use, "valid")
    res = evaluate(sub.Sales.to_numpy(), pred, sub.Open.to_numpy(),
                   label="LightGBM")

    # Same-window baseline for reference, recomputed here so the comparison
    # is never against a stale number.
    from baseline import apply_closed_rule, weekday_median_4w
    from data import VALID_START, modelling_frame
    mf = modelling_frame()
    nv = apply_closed_rule(weekday_median_4w(mf, VALID_START),
                           mf.Open.to_numpy())[mf.split.to_numpy() == "valid"]
    nsub = mf[mf.split == "valid"]
    base = evaluate(nsub.Sales.to_numpy(), nv, nsub.Open.to_numpy(),
                    label="baseline weekday_median_4w")

    print("\n" + results_table([base, res]))
    for k in ["rmse", "mape", "wape"]:
        print(f"improvement on {k.upper():5s}: "
              f"{(1 - res[k] / base[k]) * 100:5.1f}%")

    imp = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    imp["gain_pct"] = imp.gain / imp.gain.sum() * 100
    print("\ntop 20 features by gain:")
    print(imp.head(20)[["feature", "gain_pct"]].to_string(index=False,
                                                          float_format="%.2f"))
    return model, res, base, imp, use


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-store", action="store_true",
                    help="exclude the 1,115-level Store categorical")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    run(drop_store=a.drop_store, rebuild=a.rebuild)
