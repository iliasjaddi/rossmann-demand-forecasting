"""Generate a Kaggle submission for the Rossmann Store Sales competition.

Three things happen here, in order, because the last one depends on the first
two being checked rather than assumed.

1. SCORE OUR OWN TEST WINDOW WITH KAGGLE'S METRIC.
   The competition uses RMSPE, root mean squared percentage error, ignoring
   days with zero actual sales. That is not MAPE: it squares the percentage
   error before averaging, so a handful of large relative misses dominate it.
   Computing it on our sealed test window gives an honest expectation of the
   leaderboard score BEFORE submitting, which is the point of having built a
   proper holdout.

2. CHECK CALIBRATION.
   Many entries in this competition multiply the predictions by about 0.985.
   This module does not copy that number. It measures the bias on our own test
   window and calculates the correction. Under a squared percentage error, the
   best constant multiplier is always below 1 for a model that has any error.
   The shift is therefore expected from the theory as well as from practice.

3. RETRAIN ON EVERYTHING AND PREDICT.
   The served model was trained through 2015-06-13 so that the test window
   stayed clean. Kaggle's test period starts 2015-08-01, which is after all our
   labelled data, so for submission the model is refitted on every labelled day
   through 2015-07-31 at the same iteration count. More data, no leakage.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from encoding import apply_categories, load_categories
from experiment import VARIANTS, select_features
from features import feature_names
from train_lgbm import PARAMS, load_features

ROOT = Path(__file__).resolve().parent.parent
SERVE = ROOT / "serving"
CHOSEN = "no_lags"
BEST_ITER = 407


def rmspe(y_true, y_pred):
    """Kaggle's metric. Zero-sales days are ignored, per the competition rules."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = y_true > 0
    return float(np.sqrt(np.mean(((y_true[m] - y_pred[m]) / y_true[m]) ** 2)))


def main():
    df = load_features()
    drop_spec, overrides, _ = VARIANTS[CHOSEN]
    use = select_features(feature_names(df), drop_spec)
    params = {**PARAMS, **overrides}

    # ---- 1. Expected leaderboard score, from our sealed test window --------
    # model_eval.txt, not model.txt: every reported number uses the model
    # that trained on `train` only. The estimate must match the reports.
    model_eval = lgb.Booster(model_file=str(SERVE / "model_eval.txt"))
    test = df[df.split == "test"]
    pred = np.expm1(model_eval.predict(test[use], num_iteration=BEST_ITER))
    pred = np.clip(pred, 0, None)
    pred[test.Open.to_numpy() == 0] = 0.0

    base_rmspe = rmspe(test.Sales.to_numpy(), pred)
    print("=" * 66)
    print("EXPECTED LEADERBOARD SCORE")
    print("=" * 66)
    print(f"RMSPE on our sealed test window: {base_rmspe:.5f}")
    print("(competition winner scored ~0.100 on the private leaderboard)")

    # ---- 2. Calibration ---------------------------------------------------
    open_mask = test.Open.to_numpy() == 1
    yt = test.Sales.to_numpy()[open_mask]
    yp = pred[open_mask]
    ratio = yp.sum() / yt.sum()
    print(f"\nsum(predicted) / sum(actual) = {ratio:.4f}  "
          f"({'over' if ratio > 1 else 'under'}-forecasting by "
          f"{abs(1 - ratio) * 100:.2f}%)")

    factors = np.arange(0.960, 1.021, 0.005)
    scores = [(f, rmspe(yt, yp * f)) for f in factors]
    best_f, best_s = min(scores, key=lambda x: x[1])
    print("\nRMSPE against a flat multiplier:")
    for f, s in scores:
        mark = "  <-- best" if f == best_f else ""
        print(f"  x{f:.3f}   {s:.5f}{mark}")
    gain = (base_rmspe - best_s) / base_rmspe * 100
    print(f"\nbest multiplier {best_f:.3f} improves RMSPE by {gain:.2f}%")

    # Only apply a correction that is worth applying. A sub-1% gain is not
    # distinguishable from fitting noise in a single 48-day window.
    apply_factor = best_f if gain > 1.0 else 1.0
    print(f"applying multiplier: {apply_factor:.3f}"
          + ("" if apply_factor != 1.0 else "  (gain too small to trust)"))

    # ---- 3. Refit on all labelled data and predict Kaggle's test set ------
    print("\n" + "=" * 66)
    print("REFIT ON ALL LABELLED DATA")
    print("=" * 66)
    full = df[df.Open == 1]
    print(f"training rows: {len(full):,}  "
          f"through {full.Date.max().date()}  ({BEST_ITER} rounds)")
    dall = lgb.Dataset(full[use], label=np.log1p(full.Sales))
    model = lgb.train(params, dall, num_boost_round=BEST_ITER)

    serving = apply_categories(
        pd.read_parquet(SERVE / "serving_features.parquet"),
        load_categories(SERVE / "categories.json"),
    )
    kaggle = serving[serving.Sales.isna()].copy()   # the unseen period only

    # Merge on the plain integer key, never on `Store`: that column is a
    # categorical, and joining it against an int64 column silently drops the
    # dtype, which LightGBM then rejects. Re-apply the encoding afterwards
    # through the shared helper so the codes are guaranteed to match training.
    te = pd.read_csv(ROOT / "test.csv", parse_dates=["Date"])[["Id", "Store", "Date"]]
    te = te.rename(columns={"Store": "store_key"})
    kaggle = te.merge(kaggle, on=["store_key", "Date"], how="left")
    missing = int(kaggle[use].isna().all(axis=1).sum())
    if missing:
        raise SystemExit(f"{missing} test rows have no features")
    kaggle = apply_categories(kaggle, load_categories(SERVE / "categories.json"))

    p = np.expm1(model.predict(kaggle[use]))
    p = np.clip(p, 0, None) * apply_factor
    p[kaggle.Open.to_numpy() == 0] = 0.0

    sub = pd.DataFrame({"Id": kaggle.Id.astype(int), "Sales": p.round(2)})
    sub = sub.sort_values("Id")
    out = ROOT / "submission.csv"
    sub.to_csv(out, index=False)

    sample = pd.read_csv(ROOT / "sample_submission.csv")
    print(f"\nrows: {len(sub):,} (sample_submission has {len(sample):,})")
    assert len(sub) == len(sample), "row count does not match sample submission"
    assert set(sub.Id) == set(sample.Id), "Id set does not match sample submission"
    assert sub.Sales.notna().all(), "null predictions"
    print("Id set and row count match sample_submission.csv")
    print(f"\npredicted sales: mean {sub.Sales.mean():,.0f}  "
          f"zero rows {int((sub.Sales == 0).sum()):,}  "
          f"max {sub.Sales.max():,.0f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
