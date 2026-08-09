"""Load the data and split it by date.

This module builds a date-complete panel. Each store gets one row for each
calendar day in the range. The panel includes the 184 days when 180 stores
were closed for refurbishment (2014-07-01 to 2014-12-31).

The panel is necessary because pandas shifts by row position, not by date.
In the source file, the row before 2015-01-01 for a closed store is
2014-06-30. A 7-day lag would therefore read a value from six months earlier.
The panel adds empty rows for the missing days. Row distance and day distance
are then the same again. A lag that reaches into the gap returns NaN.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "train.csv"
STORE_CSV = ROOT / "store.csv"

# 48-day windows. This is the length of the Kaggle test period
# (2015-08-01 to 2015-09-17).
#
# The split is by date, not random. At forecast time you do not have future
# data. A random split lets the model train on August and test on July. It
# measures an ability that production can never use.
VALID_START = pd.Timestamp("2015-04-27")
TEST_START = pd.Timestamp("2015-06-14")
TEST_END = pd.Timestamp("2015-07-31")


def load_raw():
    """Read the two source files and join store metadata onto the daily rows."""
    df = pd.read_csv(
        TRAIN_CSV,
        parse_dates=["Date"],
        dtype={"StateHoliday": str, "SchoolHoliday": int},
    )
    store = pd.read_csv(STORE_CSV)
    return df, store


def build_panel():
    """Return one row for each (Store, Date) in the full calendar range.

    Rows that the source file does not contain get `is_gap=True` and NaN
    sales. The code never uses these rows as training targets. They keep the
    date arithmetic correct.
    """
    df, store = load_raw()

    stores = df["Store"].unique()
    dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    full_index = pd.MultiIndex.from_product([stores, dates], names=["Store", "Date"])

    panel = (
        df.set_index(["Store", "Date"])
        .reindex(full_index)
        .reset_index()
        .sort_values(["Store", "Date"])
        .reset_index(drop=True)
    )

    panel["is_gap"] = panel["Sales"].isna()

    # DayOfWeek is derivable from the date, so fill it in for gap rows too.
    # pandas: Monday=0, source file: Monday=1.
    panel["DayOfWeek"] = panel["Date"].dt.dayofweek + 1

    # Calendar flags that are known in advance are safe to carry on gap rows.
    # Sales-derived columns are deliberately left NaN.
    panel["StateHoliday"] = panel["StateHoliday"].fillna("0")
    panel["SchoolHoliday"] = panel["SchoolHoliday"].fillna(0).astype(int)

    panel = panel.merge(store, on="Store", how="left")
    return panel


def add_split(panel):
    """Label each row train / valid / test by date."""
    split = pd.Series("train", index=panel.index)
    split[panel["Date"] >= VALID_START] = "valid"
    split[panel["Date"] >= TEST_START] = "test"
    panel = panel.copy()
    panel["split"] = split
    return panel


def modelling_frame():
    """Return the panel with splits and without the gap rows.

    The code removes `Customers` on purpose. train.csv contains this column
    but the test set does not. You cannot know the number of customers six
    weeks in advance. The column would raise the training score and then fail
    in production. This is leakage.
    """
    panel = add_split(build_panel())
    panel = panel[~panel["is_gap"]].copy()
    return panel.drop(columns=["Customers"])


if __name__ == "__main__":
    p = add_split(build_panel())
    print(f"panel rows (date-complete): {len(p):,}")
    print(f"  of which gap rows:        {p.is_gap.sum():,}")
    print(f"  real observations:        {(~p.is_gap).sum():,}")
    print()
    real = p[~p.is_gap]
    print(real.groupby("split").agg(
        rows=("Sales", "size"),
        open_rows=("Open", "sum"),
        first=("Date", "min"),
        last=("Date", "max"),
    ).to_string())
