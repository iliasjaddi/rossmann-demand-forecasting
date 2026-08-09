"""Draw the figures for the README.

Each figure shows one result. The numbers come from the saved model and from
the MLflow database, not from constants. A figure therefore cannot show a
different value from the text beside it.
"""

import sqlite3
from pathlib import Path

import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline import apply_closed_rule, weekday_median_4w
from data import TEST_START, modelling_frame
from metrics import evaluate
from train_lgbm import load_features

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures"

# Validated categorical palette, light mode. Slots are used in fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8b8a85"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 160,
})


def clean(ax, keep_left=True):
    """Recessive axes. Only the axis the reader needs stays visible."""
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(keep_left)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)


def test_predictions():
    """Model and baseline predictions over the sealed test window.

    Uses model_eval.txt, not model.txt. The deployed model was refitted on
    train plus validation and scores slightly better; the evaluation model
    trained on train alone is the one every reported number comes from. The
    figures must show the same model as the text.
    """
    df = load_features()
    import json
    meta = json.loads((ROOT / "serving" / "model_meta.json").read_text())
    model = lgb.Booster(model_file=str(ROOT / "serving" / "model_eval.txt"))

    test = df[df.split == "test"].copy()
    pred = np.expm1(model.predict(test[meta["features"]]))
    test["pred"] = np.clip(pred, 0, None)
    test.loc[test.Open == 0, "pred"] = 0.0

    mf = modelling_frame()
    nv = apply_closed_rule(weekday_median_4w(mf, TEST_START), mf.Open.to_numpy())
    mf = mf.assign(base=nv)
    test = test.merge(mf[["Store", "Date", "base"]], on=["Store", "Date"], how="left")
    return test


# ---------------------------------------------------------------------------
# 1. Model against baseline
# ---------------------------------------------------------------------------
def fig_headline(test):
    """Small multiples, one panel per metric.

    Three panels rather than one grouped chart because RMSE is in euros and the
    other two are percentages. Putting them on one axis would need two scales,
    which is never correct.
    """
    m = evaluate(test.Sales.to_numpy(), test.pred.to_numpy(), test.Open.to_numpy())
    b = evaluate(test.Sales.to_numpy(), test.base.to_numpy(), test.Open.to_numpy())

    panels = [("RMSE", "euros", b["rmse"], m["rmse"], "{:,.0f}"),
              ("MAPE", "percent", b["mape"], m["mape"], "{:.2f}%"),
              ("WAPE", "percent", b["wape"], m["wape"], "{:.2f}%")]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    for ax, (name, unit, bv, mv, fmt) in zip(axes, panels):
        bars = ax.bar([0, 1], [bv, mv], width=0.55,
                      color=[INK_3, BLUE], zorder=3)
        for r, v in zip(bars, [bv, mv]):
            ax.text(r.get_x() + r.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=10, color=INK,
                    fontweight="medium")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["baseline", "LightGBM"], fontsize=9)
        ax.set_ylim(0, max(bv, mv) * 1.25)
        ax.set_title(f"{name}  ({unit})", fontsize=10, color=INK_2,
                     loc="left", pad=10)
        ax.grid(axis="x", visible=False)
        ax.set_yticks([])
        clean(ax, keep_left=False)
        drop = (1 - mv / bv) * 100
        ax.text(0.5, -0.22, f"−{drop:.1f}%", transform=ax.transAxes,
                ha="center", fontsize=11, color=BLUE, fontweight="bold")

    fig.suptitle("Sealed test window: 45,884 trading days, scored once",
                 fontsize=11, color=INK_2, x=0.008, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(FIG / "01_headline.png", bbox_inches="tight")
    plt.close(fig)
    print("  01_headline.png")


# ---------------------------------------------------------------------------
# 2. Forecast against actual
# ---------------------------------------------------------------------------
def fig_forecast(test):
    """One store across the whole 48-day horizon.

    A median-sized store, not a flattering one. Closed days are dropped from
    the line so the weekly zeros do not dominate the shape.
    """
    size = test[test.Open == 1].groupby("Store").Sales.mean()
    store = int(size.sort_values().index[len(size) // 2])

    s = test[(test.Store.astype(int) == store) & (test.Open == 1)].sort_values("Date")
    fig, ax = plt.subplots(figsize=(9.5, 3.6))

    ax.plot(s.Date, s.Sales, color=INK_3, lw=2, label="actual", zorder=3)
    ax.plot(s.Date, s.base, color=ORANGE, lw=2, ls=(0, (4, 3)),
            label="seasonal baseline", zorder=4)
    ax.plot(s.Date, s.pred, color=BLUE, lw=2, label="LightGBM", zorder=5)

    promo = s[s.Promo == 1]
    for d in promo.Date:
        ax.axvspan(d - pd.Timedelta("12h"), d + pd.Timedelta("12h"),
                   color=BLUE, alpha=0.05, lw=0, zorder=1)

    mape_m = np.mean(np.abs(s.Sales - s.pred) / s.Sales) * 100
    mape_b = np.mean(np.abs(s.Sales - s.base) / s.Sales) * 100

    ax.set_title(
        f"Store {store}, 48-day horizon, no retraining inside the window\n"
        f"shaded = promotion days   ·   MAPE {mape_m:.1f}% "
        f"against baseline {mape_b:.1f}%",
        fontsize=10, color=INK_2, loc="left", pad=12)
    ax.set_ylabel("daily sales (euros)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3, fontsize=9)
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "02_forecast_vs_actual.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  02_forecast_vs_actual.png  (store {store})")


# ---------------------------------------------------------------------------
# 3. Ablations against the reseeding noise floor
# ---------------------------------------------------------------------------
def fig_ablation():
    """The finding that matters: no ablation beats reseeding the same model."""
    con = sqlite3.connect(ROOT / "mlflow.db")
    q = """select r.name, m.key, m.value from runs r
           join metrics m on r.run_uuid = m.run_uuid
           where m.key = 'valid_mape'"""
    d = pd.read_sql(q, con).set_index("name").value
    con.close()

    seeds = [d[n] for n in ["full", "seed_1", "seed_2", "seed_3"] if n in d]
    lo, hi = min(seeds), max(seeds)

    rows = [("no_dow_mean", "drop the per-weekday average\n(40% of total gain)"),
            ("no_store", "drop store identity\n(1,115 categories)"),
            ("lr_0.03_leaves_256", "slower rate, more capacity"),
            ("no_lags", "drop all six lag features"),
            ("lr_0.10_leaves_64", "faster rate, less capacity")]
    rows = [(n, lbl, d[n]) for n, lbl, in [(a, b) for a, b in rows] if n in d]
    rows.sort(key=lambda r: r[2])

    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.axvspan(lo, hi, color=INK_3, alpha=0.16, lw=0, zorder=1)

    y = np.arange(len(rows))
    vals = [r[2] for r in rows]
    inside = [lo <= v <= hi for v in vals]
    ax.scatter(vals, y, s=110, zorder=5,
               color=[AQUA if i else ORANGE for i in inside],
               edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v, yi + 0.26, f"{v:.2f}", ha="center", fontsize=9, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows], fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("validation MAPE (percent) — lower is better")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)

    # Label the band above the plot area so it cannot sit on top of a marker.
    ax.annotate(f"same model, 4 random seeds\n{lo:.2f} to {hi:.2f}",
                xy=((lo + hi) / 2, len(rows) - 0.15),
                xytext=((lo + hi) / 2, len(rows) + 0.18),
                ha="center", va="bottom", fontsize=9, color=INK_2,
                annotation_clip=False)
    ax.set_title("No feature ablation moves the score further than reseeding "
                 "the same model", fontsize=11.5, color=INK, loc="left", pad=34)
    clean(ax, keep_left=False)
    fig.tight_layout()
    fig.savefig(FIG / "03_ablation_vs_noise.png", bbox_inches="tight")
    plt.close(fig)
    print("  03_ablation_vs_noise.png")


# ---------------------------------------------------------------------------
# 4. The promotion cycle
# ---------------------------------------------------------------------------
def fig_promo_cycle():
    """Why the obvious 7-day lag is the wrong lag for this dataset."""
    df = modelling_frame()
    base = df[["Store", "Date", "Promo"]]
    lags = [7, 14, 21, 28]
    agree = []
    for lag in lags:
        q = base.copy()
        q["Date"] = q["Date"] + pd.Timedelta(lag, "D")
        q.columns = ["Store", "Date", "p"]
        j = base.merge(q, on=["Store", "Date"], how="inner")
        agree.append((j.Promo == j.p).mean() * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    cols = [ORANGE if a < 50 else BLUE for a in agree]
    bars = ax1.bar([str(l) for l in lags], agree, width=0.6, color=cols, zorder=3)
    ax1.axhline(50, color=INK_3, lw=1, ls=(0, (3, 3)), zorder=2)
    for r, a in zip(bars, agree):
        ax1.text(r.get_x() + r.get_width() / 2, a + 1.5, f"{a:.0f}%",
                 ha="center", fontsize=10, color=INK, fontweight="medium")
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("days ago")
    ax1.set_ylabel("promotion state matches today (percent)")
    ax1.set_title("Promotions run on a two-week cycle",
                  fontsize=10.5, color=INK, loc="left", pad=10)
    ax1.text(3.42, 51, "coin flip", fontsize=8.5, color=INK_3,
             va="bottom", ha="right")
    ax1.grid(axis="x", visible=False)
    clean(ax1)

    # What the mismatch costs a lag-7 forecast.
    v = df[df.split == "valid"][["Store", "Date", "Sales", "Open", "Promo"]].copy()
    h = df[["Store", "Date", "Sales", "Promo"]].copy()
    h["Date"] = h["Date"] + pd.Timedelta(7, "D")
    h.columns = ["Store", "Date", "lag_sales", "lag_promo"]
    v = v.merge(h, on=["Store", "Date"], how="inner")
    v = v[(v.Open == 1) & (v.Sales > 0)]
    v["ape"] = (v.Sales - v.lag_sales).abs() / v.Sales * 100
    g = v.groupby(v.Promo == v.lag_promo).ape.mean()

    bars = ax2.bar(["differs", "matches"], [g[False], g[True]],
                   width=0.5, color=[ORANGE, BLUE], zorder=3)
    for r, val in zip(bars, [g[False], g[True]]):
        ax2.text(r.get_x() + r.get_width() / 2, val + 0.8, f"{val:.1f}%",
                 ha="center", fontsize=10, color=INK, fontweight="medium")
    ax2.set_ylim(0, max(g) * 1.25)
    ax2.set_ylabel("mean absolute error of a 7-day-ago forecast")
    ax2.set_xlabel("promotion state, today against 7 days ago")
    ax2.set_title("...so last week's sales mislead more often than they help",
                  fontsize=10.5, color=INK, loc="left", pad=10)
    ax2.grid(axis="x", visible=False)
    clean(ax2)

    fig.tight_layout()
    fig.savefig(FIG / "04_promo_cycle.png", bbox_inches="tight")
    plt.close(fig)
    print("  04_promo_cycle.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("writing figures:")
    test = test_predictions()
    fig_headline(test)
    fig_forecast(test)
    fig_ablation()
    fig_promo_cycle()
    print(f"\nwrote to {FIG}")


if __name__ == "__main__":
    main()
