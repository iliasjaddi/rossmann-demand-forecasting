"""Fit SARIMA on a sample of stores as a classical reference.

The module uses 15 stores, not all 1,115. The store level is the largest
source of variation, and SARIMA cannot share information between stores. 1,115
separate fits therefore give no result that a stratified sample does not
already give. They also take hours and produce many convergence failures.

The purpose is to measure the classical method, not to assume that it loses,
and to show the reason it loses. SARIMA reads the sales history of one store
and nothing else. It cannot read the promotion flag. That flag changes revenue
by 38.8% and runs on a two-week cycle.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data import VALID_START, modelling_frame
from metrics import evaluate, results_table

warnings.filterwarnings("ignore")

N_STORES = 15
SEED = 0

# (p,d,q) non-seasonal, (P,D,Q,s) seasonal with s=7 for the weekly cycle.
# d=0 because retail sales are level-stationary over 2 years (stage 1 showed
# only a mild trend, 6814 -> 7088). D=1 differences the weekly season out.
ORDER = (2, 0, 1)
SEASONAL_ORDER = (1, 1, 1, 7)


def pick_sample(df, n=N_STORES, seed=SEED):
    """Stratified by store size, so the sample is not all mid-sized stores."""
    means = df[df.Open == 1].groupby("Store").Sales.mean().sort_values()
    bins = np.array_split(means.index.to_numpy(), n)
    rng = np.random.default_rng(seed)
    return sorted(int(rng.choice(b)) for b in bins)


def prepare_series(store_df):
    """Return the history of one store as a daily series of log1p(sales).

    SARIMA needs a series with equal spacing and no holes. The code therefore
    marks the closed days as missing and interpolates them. It does not pass
    the zeros to the model. Zeros would teach the model that a fall of 100%
    each week is normal, and would damage the seasonal term.

    The code never scores the interpolated values. It sets the closed days in
    the forecast window back to 0, because the opening calendar is known in
    advance.
    """
    s = store_df.set_index("Date").Sales.astype(float)
    s = s.asfreq("D")
    s[s == 0] = np.nan
    s = np.log1p(s).interpolate(limit_direction="both")
    return s


def fit_forecast_one(store_df, cutoff, horizon):
    hist = store_df[store_df.Date < cutoff]
    s = prepare_series(hist)
    model = SARIMAX(
        s,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.forecast(steps=horizon)
    return np.expm1(fc)


def run(split="valid"):
    df = modelling_frame()
    cutoff = VALID_START
    stores = pick_sample(df)
    target = df[(df.split == split) & (df.Store.isin(stores))]
    horizon = target.Date.nunique()

    print(f"SARIMA{ORDER}x{SEASONAL_ORDER} on {len(stores)} stores, "
          f"{horizon}-day horizon\nstores: {stores}\n")

    preds, failures = [], []
    for i, st in enumerate(stores, 1):
        sdf = df[df.Store == st]
        try:
            fc = fit_forecast_one(sdf, cutoff, horizon)
            preds.append(pd.DataFrame(
                {"Store": st, "Date": fc.index, "pred_sarima": fc.to_numpy()}
            ))
            print(f"  [{i:2d}/{len(stores)}] store {st:4d} ok")
        except Exception as e:
            failures.append((st, str(e)[:60]))
            print(f"  [{i:2d}/{len(stores)}] store {st:4d} FAILED: {e!s:.60}")

    if failures:
        print(f"\n{len(failures)} convergence failures")

    pred_df = pd.concat(preds)
    m = target.merge(pred_df, on=["Store", "Date"], how="left")
    m["pred_sarima"] = m["pred_sarima"].fillna(0).clip(lower=0)
    m.loc[m.Open == 0, "pred_sarima"] = 0.0

    # Same-subset comparison: the honest baseline restricted to these stores,
    # otherwise SARIMA is being compared against a different population.
    from baseline import apply_closed_rule, weekday_median_4w
    full_pred = weekday_median_4w(df, cutoff)
    df = df.assign(pred_naive=apply_closed_rule(full_pred, df.Open.to_numpy()))
    m = m.merge(df[["Store", "Date", "pred_naive"]], on=["Store", "Date"], how="left")

    yt, op = m.Sales.to_numpy(), m.Open.to_numpy()
    rows = [
        evaluate(yt, m.pred_sarima.to_numpy(), op, label=f"SARIMA ({len(stores)} stores)"),
        evaluate(yt, m.pred_naive.to_numpy(), op, label=f"weekday_median_4w (same {len(stores)})"),
    ]
    print("\n" + results_table(rows))
    return m


if __name__ == "__main__":
    run("valid")
