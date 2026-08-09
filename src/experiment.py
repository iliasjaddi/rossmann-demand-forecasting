"""Run the experiments and record them with MLflow.

Each run stores its parameters, its validation metrics, the feature importance
table, and the model file. A statement such as "the lag features add nothing"
is then a comparison of two recorded runs.

The feature ablations are the purpose of this module. Feature importance by
gain shows how the model ordered its splits. It does not show how much each
feature adds to the accuracy. To measure that, remove the feature and train
again.

The seed runs use one configuration and different random seeds. They give the
noise level. An ablation that moves the score less than the seed runs has not
shown any effect.

    python src/experiment.py            # run all variants
    python src/experiment.py --only full no_lags
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import json
import tempfile
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from baseline import apply_closed_rule, weekday_median_4w
from data import VALID_START, modelling_frame
from features import feature_names
from metrics import evaluate
from train_lgbm import (EARLY_STOPPING, NUM_ROUNDS, PARAMS, load_features,
                        predict_full)

ROOT = Path(__file__).resolve().parent.parent
# MLflow 3.x deprecated the plain-directory store. SQLite is the supported
# local backend and is what `mlflow ui` expects; artifacts still go to disk.
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
ARTIFACT_URI = f"file://{ROOT / 'mlartifacts'}"
EXPERIMENT = "rossmann-demand"

LAG_PREFIXES = ("lag_",)
DOW_FEATURES = ("dow_mean_4", "dow_ratio")

# Each variant is (features to drop, parameter overrides, one-line rationale).
VARIANTS = {
    "full": (
        [], {},
        "All 47 features, reference run",
    ),
    "no_lags": (
        LAG_PREFIXES, {},
        "Drop the six horizon-safe lags: only 5.4% of gain, and roll_mean_28 "
        "already carries the level with less noise",
    ),
    "no_store": (
        ("Store",), {},
        "Drop the 1,115-level categorical: does dow_mean_4 already encode "
        "store level well enough?",
    ),
    "no_dow_mean": (
        DOW_FEATURES, {},
        "Drop the stage-2 baseline handed in as a feature: how much of the "
        "result rests on it?",
    ),
    "lr_0.03_leaves_256": (
        [], {"learning_rate": 0.03, "num_leaves": 256},
        "Slower learning, more capacity per tree",
    ),
    "lr_0.10_leaves_64": (
        [], {"learning_rate": 0.10, "num_leaves": 64},
        "Faster learning, less capacity: is the extra capacity earning its "
        "overfitting risk?",
    ),
    # One configuration, four random seeds. Row sampling and feature sampling
    # are random. These runs therefore give the noise level. An ablation that
    # moves the score less than these runs has shown no effect.
    "seed_1": ([], {"seed": 1, "bagging_seed": 1, "feature_fraction_seed": 1},
               "Noise floor probe, identical config"),
    "seed_2": ([], {"seed": 2, "bagging_seed": 2, "feature_fraction_seed": 2},
               "Noise floor probe, identical config"),
    "seed_3": ([], {"seed": 3, "bagging_seed": 3, "feature_fraction_seed": 3},
               "Noise floor probe, identical config"),
}


def select_features(all_feats, drop_spec):
    """Drop by exact name or by prefix."""
    keep = []
    for f in all_feats:
        if f in drop_spec or any(f.startswith(p) for p in drop_spec if p.endswith("_")):
            continue
        keep.append(f)
    return keep


def baseline_metrics():
    """Recomputed each time so the comparison is never against a stale number."""
    mf = modelling_frame()
    pred = apply_closed_rule(weekday_median_4w(mf, VALID_START), mf.Open.to_numpy())
    sub = mf[mf.split == "valid"]
    return evaluate(sub.Sales.to_numpy(), pred[mf.split.to_numpy() == "valid"],
                    sub.Open.to_numpy(), label="baseline")


def run_variant(name, df, base_metrics, all_feats):
    drop_spec, overrides, rationale = VARIANTS[name]
    use = select_features(all_feats, drop_spec)
    params = {**PARAMS, **overrides}

    tr = df[(df.split == "train") & (df.Open == 1)]
    va = df[(df.split == "valid") & (df.Open == 1)]
    dtrain = lgb.Dataset(tr[use], label=np.log1p(tr.Sales))
    dvalid = lgb.Dataset(va[use], label=np.log1p(va.Sales), reference=dtrain)

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("rationale", rationale)
        mlflow.set_tag("target_transform", "log1p")
        mlflow.set_tag("scored_on", "open days only")
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(use))
        mlflow.log_param("dropped", ",".join(drop_spec) or "none")
        mlflow.log_param("num_boost_round_max", NUM_ROUNDS)
        mlflow.log_param("early_stopping", EARLY_STOPPING)
        mlflow.log_param("train_rows", len(tr))
        mlflow.log_param("valid_window", f"{va.Date.min().date()}..{va.Date.max().date()}")

        evals = {}
        model = lgb.train(
            params, dtrain,
            num_boost_round=NUM_ROUNDS,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                lgb.record_evaluation(evals),
            ],
        )

        sub, pred = predict_full(model, df, use, "valid")
        res = evaluate(sub.Sales.to_numpy(), pred, sub.Open.to_numpy(), label=name)

        mlflow.log_metric("best_iteration", model.best_iteration)
        mlflow.log_metric("train_rmse_log", evals["train"]["rmse"][model.best_iteration - 1])
        mlflow.log_metric("valid_rmse_log", evals["valid"]["rmse"][model.best_iteration - 1])
        for k in ["rmse", "mape", "wape"]:
            mlflow.log_metric(f"valid_{k}", res[k])
            mlflow.log_metric(f"improvement_{k}_pct",
                              (1 - res[k] / base_metrics[k]) * 100)

        imp = pd.DataFrame({
            "feature": model.feature_name(),
            "gain": model.feature_importance("gain"),
        }).sort_values("gain", ascending=False)
        imp["gain_pct"] = imp.gain / imp.gain.sum() * 100

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "feature_importance.csv"
            imp.to_csv(p, index=False)
            mlflow.log_artifact(str(p))
            p2 = Path(tmp) / "features_used.json"
            p2.write_text(json.dumps(use, indent=2))
            mlflow.log_artifact(str(p2))

        mlflow.lightgbm.log_model(model, name="model")

        print(f"  {name:20s} MAPE {res['mape']:5.2f}  RMSE {res['rmse']:7.1f}  "
              f"WAPE {res['wape']:5.2f}  iters {model.best_iteration:4d}  "
              f"feats {len(use)}")
    return res


def main(only=None):
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = load_features()
    all_feats = feature_names(df)
    base = baseline_metrics()
    print(f"baseline weekday_median_4w: MAPE {base['mape']:.2f}  "
          f"RMSE {base['rmse']:.1f}  WAPE {base['wape']:.2f}\n")

    names = only or list(VARIANTS)
    results = {n: run_variant(n, df, base, all_feats) for n in names}

    print("\n" + "=" * 78)
    tbl = pd.DataFrame(results).T[["mape", "rmse", "wape"]].astype(float)
    tbl["mape_vs_full"] = tbl["mape"] - tbl.loc["full", "mape"] if "full" in tbl.index else np.nan
    tbl["improve_vs_baseline_pct"] = (1 - tbl["mape"] / base["mape"]) * 100
    print(tbl.round(2).to_string())
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(VARIANTS))
    a = ap.parse_args()
    main(a.only)
