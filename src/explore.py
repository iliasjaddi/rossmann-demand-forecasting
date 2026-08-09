"""Examine the data.

Each check in this module answers one question that a later stage must decide.
"""

import numpy as np
import pandas as pd

TRAIN = "train.csv"
STORE = "store.csv"


def load():
    df = pd.read_csv(TRAIN, parse_dates=["Date"], dtype={"StateHoliday": str})
    store = pd.read_csv(STORE)
    return df.merge(store, on="Store", how="left")


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    df = load()
    open_df = df[df.Open == 1]

    section("1. TARGET DISTRIBUTION (decides: log-transform or not)")
    s = open_df.Sales
    print(s.describe().round(1).to_string())
    print(f"\nskew(Sales)      = {s.skew():.3f}")
    print(f"skew(log1p)      = {np.log1p(s).skew():.3f}")

    section("2. MISSING DAYS PER STORE (decides: are lag features safe)")
    counts = df.groupby("Store").size()
    full = df.Date.nunique()
    gap_stores = counts[counts < full]
    print(f"calendar days in range: {full}")
    print(f"stores with a full history: {(counts == full).sum()} / {len(counts)}")
    print(f"stores with gaps:          {len(gap_stores)}")
    print(f"\nrows present, gapped stores:\n{gap_stores.describe().round(1).to_string()}")

    # Where do the gaps sit in time?
    if len(gap_stores):
        g = df[df.Store.isin(gap_stores.index)]
        present = g.groupby("Date").Store.nunique()
        missing = len(gap_stores) - present
        worst = missing[missing > 0].sort_values(ascending=False)
        print(f"\ndate window where gapped stores are absent: "
              f"{worst.index.min().date()} -> {worst.index.max().date()}")

    section("3. DAY OF WEEK (decides: how baseline handles weekly seasonality)")
    dow = df.groupby("DayOfWeek").agg(
        pct_open=("Open", "mean"),
        mean_sales_all=("Sales", "mean"),
    )
    dow["mean_sales_open"] = open_df.groupby("DayOfWeek").Sales.mean()
    dow["pct_open"] = (dow.pct_open * 100).round(1)
    print(dow.round(0).to_string())

    section("4. PROMO EFFECT (decides: does the flag carry signal at all)")
    p = open_df.groupby("Promo").Sales.agg(["count", "mean", "median"])
    lift = p.loc[1, "mean"] / p.loc[0, "mean"] - 1
    print(p.round(1).to_string())
    print(f"\nPromo lift on mean sales: {lift * 100:+.1f}%")

    p2 = open_df.groupby("Promo2").Sales.mean()
    print(f"Promo2 lift (raw, confounded): {p2.loc[1] / p2.loc[0] - 1:+.1%}")

    section("5. STORE HETEROGENEITY (decides: global model vs per-store/cluster)")
    st = open_df.groupby("Store").Sales.agg(["mean", "std"])
    st["cv"] = st["std"] / st["mean"]
    print(st["mean"].describe(percentiles=[.01, .25, .5, .75, .99]).round(0).to_string())
    print(f"\nratio p99 store mean / p01 store mean = "
          f"{st['mean'].quantile(.99) / st['mean'].quantile(.01):.1f}x")
    print(f"median within-store CV = {st['cv'].median():.3f}")

    section("6. STORE METADATA (static features available)")
    for col in ["StoreType", "Assortment", "Promo2"]:
        sub = open_df.groupby(col).Sales.agg(["mean"]).round(0)
        sub["n_stores"] = open_df.groupby(col).Store.nunique()
        print(f"\n-- {col} --\n{sub.to_string()}")
    cd = pd.read_csv(STORE).CompetitionDistance
    print(f"\nCompetitionDistance: {cd.isna().sum()} missing of {len(cd)}, "
          f"median {cd.median():.0f}m, max {cd.max():.0f}m")

    section("7. TREND OVER TIME (decides: is the holdout window representative)")
    m = open_df.set_index("Date").groupby(pd.Grouper(freq="MS")).Sales.mean()
    print(m.round(0).tail(18).to_string())
    yr = open_df.groupby(open_df.Date.dt.year).Sales.mean()
    print(f"\nyearly mean (open days):\n{yr.round(0).to_string()}")

    section("8. PROPOSED SPLIT (48-day windows, mirrors Kaggle's test horizon)")
    end = df.Date.max()
    test_start = end - pd.Timedelta(days=47)
    val_start = test_start - pd.Timedelta(days=48)
    for name, lo, hi in [
        ("train", df.Date.min(), val_start - pd.Timedelta(days=1)),
        ("valid", val_start, test_start - pd.Timedelta(days=1)),
        ("test", test_start, end),
    ]:
        n = df[(df.Date >= lo) & (df.Date <= hi)]
        print(f"{name:6s} {lo.date()} -> {hi.date()}  "
              f"{len(n):>7,} rows  ({(n.Open == 1).sum():>7,} open)")


if __name__ == "__main__":
    main()
