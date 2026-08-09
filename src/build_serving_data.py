"""Build the feature table that the API reads.

The module does two things.

1. It extends the panel past the end of train.csv with test.csv. That file
   gives the published calendar (Open, Promo, StateHoliday, SchoolHoliday) for
   2015-08-01 to 2015-09-17. It gives no sales. These are future days that no
   step in this project has seen. The service can therefore demonstrate on
   unseen dates instead of on training data.

2. It calculates all 41 features for the serving window and writes them to a
   parquet file. The container includes this file.

Step 2 is possible because each sales feature has a lag of 48 days or more. To
predict date d, the code reads sales from d-48 or earlier. History ends on
2015-07-31, so the last date the service can predict is 2015-09-17. This is
also the last date of the Kaggle test set.
"""

import json
from pathlib import Path

import pandas as pd

from data import STORE_CSV, TRAIN_CSV
from encoding import apply_categories, save_categories, training_categories
from features import (add_calendar, add_competition, add_promo2,
                      add_promo_context, add_sales_history)

ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT / "test.csv"
SERVE_DIR = ROOT / "serving"

SERVE_FROM = pd.Timestamp("2015-06-14")   # start of our sealed test window
SERVE_TO = pd.Timestamp("2015-09-17")     # last date the 48-day rule allows


def load_extended():
    """train.csv rows plus test.csv calendar rows, as one continuous history."""
    tr = pd.read_csv(TRAIN_CSV, parse_dates=["Date"],
                     dtype={"StateHoliday": str, "SchoolHoliday": int})
    tr = tr.drop(columns=["Customers"])

    te = pd.read_csv(TEST_CSV, parse_dates=["Date"],
                     dtype={"StateHoliday": str, "SchoolHoliday": int})
    te = te.drop(columns=["Id"])
    # 11 rows in test.csv have no Open value. Kaggle's convention, and the
    # sensible default, is that a store trades unless told otherwise.
    te["Open"] = te["Open"].fillna(1).astype(int)
    te["Sales"] = pd.NA          # genuinely unknown, not zero

    both = pd.concat([tr, te], ignore_index=True)
    both["Sales"] = pd.to_numeric(both["Sales"], errors="coerce")
    return both.sort_values(["Store", "Date"]).reset_index(drop=True)


def build_panel_extended(df):
    """Same date-complete reindex as data.build_panel, over the longer range."""
    stores = df["Store"].unique()
    dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    idx = pd.MultiIndex.from_product([stores, dates], names=["Store", "Date"])

    panel = (df.set_index(["Store", "Date"]).reindex(idx).reset_index()
             .sort_values(["Store", "Date"]).reset_index(drop=True))

    # A row is a real calendar day for this store if the source files had it.
    have = set(zip(df["Store"], df["Date"]))
    panel["is_gap"] = [(s, d) not in have
                       for s, d in zip(panel["Store"], panel["Date"])]

    panel["DayOfWeek"] = panel["Date"].dt.dayofweek + 1
    panel["StateHoliday"] = panel["StateHoliday"].fillna("0")
    panel["SchoolHoliday"] = panel["SchoolHoliday"].fillna(0).astype(int)
    panel["Open"] = panel["Open"].fillna(0).astype(int)
    panel["Promo"] = panel["Promo"].fillna(0).astype(int)

    store = pd.read_csv(STORE_CSV)
    return panel.merge(store, on="Store", how="left")


def main():
    SERVE_DIR.mkdir(exist_ok=True)
    panel = build_panel_extended(load_extended())
    print(f"extended panel: {len(panel):,} rows, "
          f"{panel.Date.min().date()} -> {panel.Date.max().date()}")

    panel = add_calendar(panel)
    panel = add_promo_context(panel)
    panel = add_promo2(panel)
    panel = add_competition(panel)
    panel = add_sales_history(panel)
    panel = panel[~panel["is_gap"]].copy()

    meta = json.loads((SERVE_DIR / "model_meta.json").read_text())
    feats = meta["features"]

    serve = panel[(panel.Date >= SERVE_FROM) & (panel.Date <= SERVE_TO)].copy()
    serve["store_key"] = serve["Store"].astype(int)   # plain key for lookups
    keep = ["store_key", "Store", "Date", "Open", "Promo", "Sales"] + feats
    serve = serve[list(dict.fromkeys(keep))]

    # Freeze the exact category lists training used. Applying them is a check
    # here, not the delivery mechanism: parquet does not round-trip the
    # category dtype, so the API re-applies them through the same shared
    # helper. apply_categories raises rather than producing silent NaNs if any
    # value falls outside the training vocabulary.
    cats = {c: v for c, v in training_categories().items() if c in feats}
    apply_categories(serve, cats, strict=True)
    save_categories(cats, SERVE_DIR / "categories.json")

    out = SERVE_DIR / "serving_features.parquet"
    serve.to_parquet(out, index=False)

    known = serve.Sales.notna()
    print(f"\nserving window: {SERVE_FROM.date()} -> {SERVE_TO.date()}")
    print(f"  rows:                {len(serve):,}")
    print(f"  distinct stores:     {serve.Store.nunique():,}")
    print(f"  with known sales:    {known.sum():,} (our sealed test window)")
    print(f"  truly unseen:        {(~known).sum():,} (Kaggle test period)")
    print(f"  features frozen:     {len(feats)}")
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {SERVE_DIR / 'categories.json'}")

    missing = serve[feats].isna().mean()
    missing = missing[missing > 0.5]
    if len(missing):
        print("\nWARNING: features more than 50% missing in serving window:")
        print((missing * 100).round(1).to_string())


if __name__ == "__main__":
    main()
