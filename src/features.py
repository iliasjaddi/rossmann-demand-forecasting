"""Build the model features.

One rule controls this module. The forecast horizon is 48 days. At prediction
time you cannot use any sales value from inside that horizon. Each feature
that comes from sales therefore uses a lag of 48 days or more.

  horizon day 48 -> lag_49 reads cutoff - 2   (known)
  horizon day  1 -> lag_49 reads cutoff - 49  (known)

One model can then predict all 48 days in one pass. It does not feed its own
predictions back, so the errors do not accumulate.

Promo, StateHoliday, SchoolHoliday and Open are published schedules, not
results. The Kaggle test file contains them. You can therefore use them at any
horizon.

The code builds the features on the date-complete panel from data.py. A
groupby().shift(n) then moves exactly n calendar days. On the source file it
would not: for 180 stores the shift would cross a six-month gap.
"""

import numpy as np
import pandas as pd

from data import add_split, build_panel

HORIZON = 48

# Each lag is a multiple of 7, so it falls on the same weekday.
# 56 and 70 are also multiples of 14, so they fall in the same promotion week.
# 49 and 63 fall in the opposite promotion week.
# The list contains both types on purpose. With the promotion flags, the model
# can use the difference between them.
LAGS = [49, 56, 63, 70, 77, 91]

# Rolling windows, all computed on a series already shifted by 48 days.
ROLL_WINDOWS = [7, 14, 28, 91]

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def add_calendar(df):
    """Add the features that come from the date. All are known in advance.

    DayOfWeek and month stay as integers. The code does not use sine and
    cosine encoding. That encoding lets a linear model or a neural network see
    that Monday follows Sunday. A boosted tree splits on thresholds and can
    isolate one day with two splits. Sine and cosine would make the tree
    approximate a step with a curve. This costs splits and gives no benefit.
    """
    d = df["Date"]
    df["day"] = d.dt.day
    df["month"] = d.dt.month
    df["year"] = d.dt.year
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["day_of_year"] = d.dt.dayofyear

    # Sales rise at the start of the month, after payday, and again at the
    # end. A tree can find this from `day` alone. A direct flag makes the
    # effect one split instead of several.
    df["days_to_month_end"] = d.dt.days_in_month - d.dt.day
    df["is_month_start"] = (df["day"] <= 5).astype(int)
    df["is_month_end"] = (df["days_to_month_end"] <= 5).astype(int)
    return df


def add_promo_context(df):
    """Add promotion features other than the same-day flag.

    The promotion calendar is published in advance. The code can therefore use
    the days before and after. Promotions run on a two-week cycle and change
    sales by 38.8%. This block is the most useful one in the file.
    """
    g = df.groupby("Store", sort=False)["Promo"]

    # The promotion state one day before and one day after. Sales increase on
    # the first day of a promotion and decrease after the last day.
    df["promo_prev_day"] = g.shift(1).fillna(0).astype(int)
    df["promo_next_day"] = g.shift(-1).fillna(0).astype(int)
    df["promo_first_day"] = ((df["Promo"] == 1) & (df["promo_prev_day"] == 0)).astype(int)
    df["promo_last_day"] = ((df["Promo"] == 1) & (df["promo_next_day"] == 0)).astype(int)

    # The promotion state one week and two weeks ago. The model can then read
    # the position in the two-week cycle directly.
    df["promo_lag7"] = g.shift(7).fillna(0).astype(int)
    df["promo_lag14"] = g.shift(14).fillna(0).astype(int)

    # How many promo days in the trailing fortnight (0-14).
    df["promo_count_14d"] = (
        g.rolling(14, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0)
    )
    return df


def add_promo2(df):
    """Decode Promo2. It is a continuous programme, not a daily flag.

    A store joins from a given calendar week. The promotion then restarts in
    the months that PromoInterval names.

    The raw Promo2 column has almost no value. It correlates negatively with
    sales because the smaller stores joined the programme. What the model
    needs is whether the programme is active on this date, and for how long.
    """
    start = pd.to_datetime(
        df["Promo2SinceYear"].astype("Int64").astype(str)
        + "-" + df["Promo2SinceWeek"].astype("Int64").astype(str) + "-1",
        format="%Y-%W-%w", errors="coerce",
    )
    df["promo2_active"] = (
        (df["Promo2"] == 1) & (df["Date"] >= start)
    ).astype(int)
    df["promo2_weeks_running"] = np.where(
        df["promo2_active"] == 1,
        (df["Date"] - start).dt.days / 7.0,
        -1.0,
    )

    # Is the current month one of the months this store's Promo2 restarts in?
    month_name = df["month"].map(lambda m: MONTH_ABBR[m - 1])
    interval = df["PromoInterval"].fillna("")
    df["promo2_restart_month"] = (
        [1 if mn in iv.split(",") else 0 for mn, iv in zip(month_name, interval)]
    )
    df["promo2_restart_month"] *= df["promo2_active"]
    return df


def add_competition(df):
    """Add the competitor distance and the age of the competitor.

    The code takes the logarithm of the distance. The difference between 100 m
    and 600 m is large. The difference between 70,000 m and 70,500 m is not. A
    tree can find this with enough splits. The logarithm makes it one split.

    3 of the 1,115 stores have no distance. This most probably means no
    competitor is near. The code therefore fills a large value, not the median.
    """
    dist = df["CompetitionDistance"]
    df["competition_distance_log"] = np.log1p(dist.fillna(dist.max() * 2))
    df["competition_distance_missing"] = dist.isna().astype(int)

    opened = pd.to_datetime(
        df["CompetitionOpenSinceYear"].astype("Int64").astype(str)
        + "-" + df["CompetitionOpenSinceMonth"].astype("Int64").astype(str) + "-01",
        errors="coerce",
    )
    months = (df["Date"] - opened).dt.days / 30.44
    # Negative means the competitor had not opened yet on this date.
    df["competition_open_months"] = months.fillna(-999).clip(-999, 240)
    return df


def add_sales_history(df):
    """Add the lags and the rolling statistics. Each uses a lag of 48 days.

    The code makes two choices here.

    1. It calculates the statistics on trading days only. Closed days are
       exactly 0. A rolling mean that includes them measures the sales
       multiplied by the open rate. These are two different quantities. The
       open rate is a separate feature.

    2. It includes the level (mean and median) and the spread (standard
       deviation). A store with a mean of 7,000 and a spread of 500 is a
       different problem from a store with a mean of 7,000 and a spread of
       3,000. The model can use the spread to decide how much to trust the
       level.
    """
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # Sales on trading days only. Closed days become NaN. The rolling
    # statistics then skip them instead of moving the result toward zero.
    df["_sales_open"] = df["Sales"].where(df["Open"] == 1)

    g = df.groupby("Store", sort=False)["_sales_open"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)

    # The base series for each rolling window. It is already 48 days old, so
    # any window built on it is valid for the whole horizon.
    df["_shift48"] = g.shift(HORIZON)
    gs = df.groupby("Store", sort=False)["_shift48"]
    for w in ROLL_WINDOWS:
        mp = max(2, w // 3)   # tolerate the refurbishment gap
        df[f"roll_mean_{w}"] = (
            gs.rolling(w, min_periods=mp).mean().reset_index(level=0, drop=True)
        )
        df[f"roll_std_{w}"] = (
            gs.rolling(w, min_periods=mp).std().reset_index(level=0, drop=True)
        )
    df["roll_median_28"] = (
        gs.rolling(28, min_periods=9).median().reset_index(level=0, drop=True)
    )

    # Trend: is the store growing or shrinking, as of the cutoff.
    df["trend_28_over_91"] = df["roll_mean_28"] / df["roll_mean_91"]

    # How often the store traded. This shows refurbishment and unusual
    # opening patterns without the use of any sales value.
    df["_open48"] = df.groupby("Store", sort=False)["Open"].shift(HORIZON)
    df["open_rate_28"] = (
        df.groupby("Store", sort=False)["_open48"]
        .rolling(28, min_periods=9).mean().reset_index(level=0, drop=True)
    )

    # Same-weekday history. This is the seasonal baseline as a feature. It
    # scored 17.16% MAPE on its own. It is the mean of this weekday over the 4
    # same-weekday values before the 48-day cutoff.
    df["dow_mean_4"] = (
        df.groupby(["Store", "DayOfWeek"], sort=False)["_sales_open"]
        .transform(lambda s: s.shift(7).rolling(4, min_periods=2).mean())
    )
    # Relative shape of this weekday vs the store's overall level.
    df["dow_ratio"] = df["dow_mean_4"] / df["roll_mean_28"]

    return df.drop(columns=["_sales_open", "_shift48", "_open48"])


CATEGORICAL = ["Store", "StoreType", "Assortment", "StateHoliday", "DayOfWeek"]

DROP = [
    "Date", "Sales", "Open", "is_gap", "split", "Customers",
    "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
]


def build(verbose=True):
    """Full feature frame, gap rows removed, splits labelled."""
    panel = add_split(build_panel())

    panel = add_calendar(panel)
    panel = add_promo_context(panel)
    panel = add_promo2(panel)
    panel = add_competition(panel)
    panel = add_sales_history(panel)

    # The gap rows only kept the date arithmetic correct. Remove them now.
    panel = panel[~panel["is_gap"]].copy()

    for c in CATEGORICAL:
        panel[c] = panel[c].astype("category")

    if verbose:
        feats = feature_names(panel)
        print(f"rows: {len(panel):,}   features: {len(feats)}")
        miss = panel[feats].isna().mean().sort_values(ascending=False)
        miss = miss[miss > 0]
        if len(miss):
            print("\nfeatures with missing values (LightGBM handles NaN natively):")
            print((miss * 100).round(2).to_string())
    return panel


def feature_names(panel):
    return [c for c in panel.columns if c not in DROP]


if __name__ == "__main__":
    p = build()
    print("\nfeature list:")
    for i, f in enumerate(feature_names(p), 1):
        print(f"  {i:2d}. {f}")
    out = "features.parquet"
    p.to_parquet(out, index=False)
    print(f"\nwrote {out}")
