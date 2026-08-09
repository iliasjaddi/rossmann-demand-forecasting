"""Test the features for time leakage.

The statement to test: no feature that predicts a validation day uses a sales
value from inside the validation window.

The method: build the sales features a second time on a copy of the panel. In
the copy, every sales value from VALID_START onward is removed. The copy
therefore contains only what is known at the cutoff. If the validation rows
are identical in both versions, no future value reached them.

This test is stronger than a manual check of the shift() arguments. An error
of one day in a rolling window is not visible on the page. It also raises the
score by a large amount.
"""

import numpy as np
import pandas as pd

from data import VALID_START, add_split, build_panel
from features import (add_calendar, add_competition, add_promo2,
                      add_promo_context, add_sales_history)

SALES_DERIVED_PREFIXES = ("lag_", "roll_", "dow_mean", "dow_ratio",
                          "trend_", "open_rate_")


def pipeline(panel):
    panel = add_calendar(panel)
    panel = add_promo_context(panel)
    panel = add_promo2(panel)
    panel = add_competition(panel)
    return add_sales_history(panel)


def main():
    base = add_split(build_panel())

    full = pipeline(base.copy())

    # The counterfactual: standing at the cutoff, sales from VALID_START
    # onward simply do not exist yet.
    blind = base.copy()
    future = blind["Date"] >= VALID_START
    blind.loc[future, "Sales"] = np.nan
    blind = pipeline(blind)

    cols = [c for c in full.columns if c.startswith(SALES_DERIVED_PREFIXES)]
    v_full = full[full["split"] == "valid"].sort_values(["Store", "Date"])
    v_blind = blind[blind["split"] == "valid"].sort_values(["Store", "Date"])

    print(f"validation rows compared: {len(v_full):,}")
    print(f"sales-derived features:   {len(cols)}\n")

    bad = []
    for c in cols:
        a, b = v_full[c].to_numpy(), v_blind[c].to_numpy()
        same = np.isclose(a, b, rtol=1e-9, equal_nan=True)
        n_diff = int((~same).sum())
        if n_diff:
            bad.append((c, n_diff, n_diff / len(a) * 100))

    if not bad:
        print("PASS: every sales-derived feature is identical with and")
        print("      without access to validation-window sales.")
    else:
        print("FAIL: these features change when future sales are hidden,")
        print("      which means they leak:\n")
        for c, n, pct in sorted(bad, key=lambda x: -x[1]):
            print(f"  {c:22s} {n:>7,} rows differ ({pct:.2f}%)")

    # Second check: the target itself must never appear among the inputs.
    corr_check = v_full[["Sales"] + cols].corr()["Sales"].drop("Sales")
    print(f"\nhighest correlation of any feature with same-day Sales: "
          f"{corr_check.abs().max():.3f} ({corr_check.abs().idxmax()})")
    print("(a value near 1.0 would mean the target leaked in directly)")


if __name__ == "__main__":
    main()
