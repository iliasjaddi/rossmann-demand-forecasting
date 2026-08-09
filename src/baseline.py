"""Calculate the baseline forecasts.

These give the score that the boosted model must beat. A model that improves
on nothing is not a result.

The file contains three baselines. The difference between them is the most
important part of this module.

  1. lag7_rolling      -- use the sales from 7 days ago, including days inside
                          the forecast window. This is not valid for a 6-week
                          horizon. It is here to show how much it changes the
                          score.
  2. last_week_tiled   -- take the last week of history before the forecast,
                          then repeat it for all 7 weeks. This is valid.
  3. weekday_median_4w -- for each store, take the median of each weekday over
                          the last 4 weeks. This is valid, and one unusual
                          week changes it less.

Each baseline predicts 0 for a closed day. The opening calendar is known in
advance, so the code can use `Open` at prediction time. It cannot use `Sales`
from inside the horizon.
"""

import numpy as np
import pandas as pd

from data import TEST_END, TEST_START, VALID_START, modelling_frame
from metrics import evaluate, results_table

HORIZON_DAYS = 48


def _window(df, split):
    return df[df["split"] == split]


def naive_lag7_rolling(df, cutoff):
    """Return the sales from exactly 7 calendar days earlier.

    The code joins on the date, not on the row position. The 2014 gap
    therefore gives NaN, not a value from six months earlier.

    This is not a valid baseline for a 6-week forecast. To predict day 40 of
    the horizon it reads day 33. Day 33 is inside the horizon and is unknown
    at forecast time. The function is here as a demonstration.
    """
    hist = df[["Store", "Date", "Sales"]].copy()
    hist["Date"] = hist["Date"] + pd.Timedelta(7, "D")
    hist = hist.rename(columns={"Sales": "pred"})
    out = df.merge(hist, on=["Store", "Date"], how="left")
    return out["pred"].to_numpy()


def _history_before(df, cutoff):
    return df[df["Date"] < cutoff]


def last_week_tiled(df, cutoff):
    """Repeat the last full week of history across the whole horizon.

    The function uses only data before `cutoff`. It is therefore a valid
    48-day forecast. Every value it needs is known on the day of the forecast.
    """
    hist = _history_before(df, cutoff)
    last7 = hist[hist["Date"] >= cutoff - pd.Timedelta(7, "D")]

    # One value per (store, weekday) from that final week.
    template = (
        last7.groupby(["Store", "DayOfWeek"])["Sales"]
        .mean()
        .rename("pred")
        .reset_index()
    )
    out = df.merge(template, on=["Store", "DayOfWeek"], how="left")
    return out["pred"].to_numpy()


def weekday_median_4w(df, cutoff, weeks=4):
    """Return the median sales for each (store, weekday) over `weeks` weeks.

    The function uses trading days only. One closed day therefore does not
    move the result toward zero. If a weekday never occurs as a trading day in
    the window, the function uses the median for the store. This applies
    mostly to Sundays.
    """
    hist = _history_before(df, cutoff)
    look = hist[hist["Date"] >= cutoff - pd.Timedelta(7 * weeks, "D")]
    open_look = look[look["Open"] == 1]

    template = (
        open_look.groupby(["Store", "DayOfWeek"])["Sales"]
        .median()
        .rename("pred")
        .reset_index()
    )
    fallback = (
        hist[hist["Open"] == 1].groupby("Store")["Sales"].median().rename("fb")
    )

    out = df.merge(template, on=["Store", "DayOfWeek"], how="left")
    out = out.merge(fallback, on="Store", how="left")
    return out["pred"].fillna(out["fb"]).to_numpy()


def apply_closed_rule(pred, is_open):
    """Set the prediction to 0 for each closed day.

    A closed store sells nothing. This is a fact, not a forecast.
    """
    pred = np.asarray(pred, dtype=float).copy()
    pred[np.asarray(is_open) == 0] = 0.0
    return np.nan_to_num(pred, nan=0.0)


def run(split="valid"):
    df = modelling_frame()
    cutoff = VALID_START if split == "valid" else TEST_START
    target = _window(df, split)

    print(f"Scoring on {split}: {target.Date.min().date()} -> "
          f"{target.Date.max().date()}, {len(target):,} rows, "
          f"{int(target.Open.sum()):,} open\n")

    methods = {
        "lag7_rolling (INVALID, leaks)": naive_lag7_rolling,
        "last_week_tiled": last_week_tiled,
        "weekday_median_4w": weekday_median_4w,
    }

    rows = []
    preds = {}
    in_split = df["split"].to_numpy() == split
    for name, fn in methods.items():
        # Every method is computed against the FULL frame then sliced, so the
        # first days of the horizon can still see the history behind them.
        raw = fn(df, cutoff)[in_split]
        pred = apply_closed_rule(raw, target["Open"].to_numpy())
        preds[name] = pred
        rows.append(evaluate(target["Sales"].to_numpy(), pred,
                             target["Open"].to_numpy(), label=name))

    print(results_table(rows))
    return target, preds


if __name__ == "__main__":
    run("valid")
