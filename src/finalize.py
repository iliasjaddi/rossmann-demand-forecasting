"""Score the sealed test window, then write the files for serving.

No step before this one used the test window (2015-06-14 to 2015-07-31). No
parameter was tuned on it. Early stopping did not read it. No feature was
selected with it. This module scores it one time.

The module writes two models.

  1. The evaluation model. It trains on `train`, stops early on `valid`, and
     scores on `test`. Every reported number comes from this model.
  2. The serving model. It uses the same configuration and the same number of
     rounds, but trains on train and valid together. The deployed file then
     uses every labelled day up to the cutoff. This model cannot be scored
     without leakage, so the reported numbers never come from it.
"""

import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from baseline import apply_closed_rule, weekday_median_4w
from data import TEST_START, modelling_frame
from experiment import ARTIFACT_URI, EXPERIMENT, TRACKING_URI, VARIANTS, select_features
from features import feature_names
from metrics import evaluate, results_table
from train_lgbm import EARLY_STOPPING, NUM_ROUNDS, PARAMS, load_features

ROOT = Path(__file__).resolve().parent.parent
SERVE_DIR = ROOT / "serving"
CHOSEN = "no_lags"   # statistically tied with `full`, 6 fewer features


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = load_features()
    drop_spec, overrides, _ = VARIANTS[CHOSEN]
    use = select_features(feature_names(df), drop_spec)
    params = {**PARAMS, **overrides}

    tr = df[(df.split == "train") & (df.Open == 1)]
    va = df[(df.split == "valid") & (df.Open == 1)]

    # ---- 1. Honest evaluation model -------------------------------------
    dtrain = lgb.Dataset(tr[use], label=np.log1p(tr.Sales))
    dvalid = lgb.Dataset(va[use], label=np.log1p(va.Sales), reference=dtrain)
    model = lgb.train(
        params, dtrain, num_boost_round=NUM_ROUNDS,
        valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    best_iter = model.best_iteration
    print(f"config: {CHOSEN}   features: {len(use)}   best_iteration: {best_iter}\n")

    rows = []
    for split in ["valid", "test"]:
        sub = df[df.split == split]
        pred = np.expm1(model.predict(sub[use], num_iteration=best_iter))
        pred = np.clip(pred, 0, None)
        pred[sub.Open.to_numpy() == 0] = 0.0
        rows.append(evaluate(sub.Sales.to_numpy(), pred, sub.Open.to_numpy(),
                             label=f"LightGBM {split}"))

    # Baseline on the same windows, recomputed at each window's own cutoff.
    mf = modelling_frame()
    from data import VALID_START
    for split, cutoff in [("valid", VALID_START), ("test", TEST_START)]:
        p = apply_closed_rule(weekday_median_4w(mf, cutoff), mf.Open.to_numpy())
        sub = mf[mf.split == split]
        rows.append(evaluate(sub.Sales.to_numpy(),
                             p[mf.split.to_numpy() == split],
                             sub.Open.to_numpy(), label=f"baseline {split}"))

    order = ["baseline valid", "LightGBM valid", "baseline test", "LightGBM test"]
    rows = sorted(rows, key=lambda r: order.index(r["label"]))
    print(results_table(rows))

    bt = next(r for r in rows if r["label"] == "baseline test")
    mt = next(r for r in rows if r["label"] == "LightGBM test")
    print("\nHEADLINE (test window, never used for any decision):")
    for k in ["rmse", "mape", "wape"]:
        print(f"  {k.upper():5s} {bt[k]:8.2f} -> {mt[k]:7.2f}   "
              f"improvement {(1 - mt[k] / bt[k]) * 100:5.1f}%")

    with mlflow.start_run(run_name="final_test_eval"):
        mlflow.set_tag("note", "sealed test window, scored once")
        mlflow.log_params({**params, "config": CHOSEN, "n_features": len(use)})
        mlflow.log_metric("best_iteration", best_iter)
        for r in rows:
            tag = r["label"].replace(" ", "_").lower()
            for k in ["rmse", "mape", "wape"]:
                mlflow.log_metric(f"{tag}_{k}", r[k])
        for k in ["rmse", "mape", "wape"]:
            mlflow.log_metric(f"headline_improvement_{k}_pct",
                              (1 - mt[k] / bt[k]) * 100)

    # ---- 2. Serving model, refit on train + valid ------------------------
    both = df[(df.split.isin(["train", "valid"])) & (df.Open == 1)]
    dall = lgb.Dataset(both[use], label=np.log1p(both.Sales))
    serve_model = lgb.train(params, dall, num_boost_round=best_iter)

    SERVE_DIR.mkdir(exist_ok=True)
    serve_model.save_model(str(SERVE_DIR / "model.txt"))
    # The evaluation model is saved too. Every reported number and every README
    # figure comes from this one, so a figure cannot quietly show a different
    # model from the text beside it.
    model.save_model(str(SERVE_DIR / "model_eval.txt"))
    (SERVE_DIR / "model_meta.json").write_text(json.dumps({
        "config": CHOSEN,
        "features": use,
        "categorical": [c for c in ["Store", "StoreType", "Assortment",
                                    "StateHoliday", "DayOfWeek"] if c in use],
        "num_boost_round": best_iter,
        "target_transform": "log1p",
        "trained_through": str(both.Date.max().date()),
        "test_metrics": {k: mt[k] for k in ["rmse", "mape", "wape"]},
        "baseline_test_metrics": {k: bt[k] for k in ["rmse", "mape", "wape"]},
    }, indent=2))
    print(f"\nwrote {SERVE_DIR/'model.txt'} "
          f"(refit on train+valid, {best_iter} rounds, "
          f"through {both.Date.max().date()})")


if __name__ == "__main__":
    main()
