"""Serve the forecasts over HTTP.

    uvicorn api:app --reload --port 8000    (from src/)
    open http://localhost:8000/docs

The service keeps no state between requests. At startup it loads one LightGBM
model and one prepared feature table. Each request is one lookup and one
forward pass.

This design works because each feature is at least 48 days old. No request can
need a value from inside its own horizon. A recursive design would need 48
model calls in sequence for each request, and a place to hold the
intermediate values.
"""

import json
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from encoding import apply_categories, load_categories

SERVE_DIR = Path(__file__).resolve().parent.parent / "serving"

MODEL = lgb.Booster(model_file=str(SERVE_DIR / "model.txt"))
META = json.loads((SERVE_DIR / "model_meta.json").read_text())
FEATURES = META["features"]

# Categories come from the frozen file through the shared helper, never
# re-derived here. Re-deriving them is precisely how the encoding drifts out
# of sync with the model. `store_key` stays a plain int column, because
# `Store` is a categorical and comparing a categorical to an int matches
# nothing at all.
FRAME = apply_categories(
    pd.read_parquet(SERVE_DIR / "serving_features.parquet"),
    load_categories(SERVE_DIR / "categories.json"),
)

TEMPLATE = (Path(__file__).resolve().parent / "landing.html").read_text()

SERVE_FROM = FRAME.Date.min().date()
SERVE_TO = FRAME.Date.max().date()
STORE_IDS = sorted(FRAME.store_key.unique().tolist())
STORE_RANGE = (FRAME.groupby("store_key")["Date"]
               .agg(["min", "max"]).to_dict("index"))
LAST_TRAINED = META["trained_through"]

app = FastAPI(
    title="Rossmann demand forecast",
    version="1.0.0",
    description=(
        "Daily sales forecasts per store, up to a 48-day horizon. "
        f"Model: LightGBM on log1p(sales), {META['num_boost_round']} rounds, "
        f"{len(FEATURES)} features. Test-window MAPE "
        f"{META['test_metrics']['mape']:.2f}% against a seasonal-median "
        f"baseline at {META['baseline_test_metrics']['mape']:.2f}%."
    ),
)


class DayForecast(BaseModel):
    date: date
    day_of_week: int
    open: bool
    promo: bool
    predicted_sales: float
    actual_sales: float | None = Field(
        None, description="Present only for dates whose sales are in the "
                          "public dataset; null for genuinely unseen dates."
    )
    absolute_percentage_error: float | None = None


class Forecast(BaseModel):
    store_id: int
    start_date: date
    end_date: date
    n_days: int
    n_trading_days: int
    total_predicted_sales: float
    mean_predicted_sales_per_trading_day: float
    mape_vs_actual: float | None = Field(
        None, description="Only computable where actual sales are known."
    )
    forecast: list[DayForecast]


class PredictRequest(BaseModel):
    store_id: int = Field(..., examples=[262])
    start_date: date = Field(..., examples=["2015-08-01"])
    end_date: date = Field(..., examples=["2015-08-14"])


def _predict(store_id: int, start: date, end: date) -> Forecast:
    if store_id not in STORE_IDS:
        raise HTTPException(404, f"unknown store {store_id}")
    if end < start:
        raise HTTPException(400, "end_date is before start_date")
    if start < SERVE_FROM or end > SERVE_TO:
        raise HTTPException(
            400,
            f"servable range is {SERVE_FROM} to {SERVE_TO}. The upper bound is "
            f"48 days past the last day of sales history the model was built "
            f"on; beyond it the features would need data that does not exist.",
        )

    sub = FRAME[(FRAME.store_key == store_id)
                & (FRAME.Date >= pd.Timestamp(start))
                & (FRAME.Date <= pd.Timestamp(end))].sort_values("Date")
    if sub.empty:
        r = STORE_RANGE[store_id]
        raise HTTPException(
            404,
            f"store {store_id} has no calendar rows between {start} and {end}. "
            f"Its servable range is {r['min'].date()} to {r['max'].date()}. "
            f"Only 856 of the 1,115 stores appear in Kaggle's held-out period "
            f"(2015-08-01 onward); the rest can only be forecast up to "
            f"2015-07-31.",
        )

    pred = np.expm1(MODEL.predict(sub[FEATURES]))
    pred = np.clip(pred, 0, None)
    pred[sub.Open.to_numpy() == 0] = 0.0     # a closed store sells nothing

    days, errs = [], []
    for (_, row), p in zip(sub.iterrows(), pred):
        actual = None if pd.isna(row.Sales) else float(row.Sales)
        ape = None
        if actual is not None and actual > 0:
            ape = abs(actual - p) / actual * 100
            errs.append(ape)
        days.append(DayForecast(
            date=row.Date.date(),
            day_of_week=int(row.Date.dayofweek) + 1,
            open=bool(row.Open),
            promo=bool(row.Promo),
            predicted_sales=round(float(p), 2),
            actual_sales=actual,
            absolute_percentage_error=None if ape is None else round(ape, 2),
        ))

    trading = int(sub.Open.sum())
    total = float(pred.sum())
    return Forecast(
        store_id=store_id,
        start_date=start,
        end_date=end,
        n_days=len(days),
        n_trading_days=trading,
        total_predicted_sales=round(total, 2),
        mean_predicted_sales_per_trading_day=round(total / max(trading, 1), 2),
        mape_vs_actual=round(float(np.mean(errs)), 2) if errs else None,
        forecast=days,
    )


@app.post("/predict", response_model=Forecast, summary="Forecast a date range")
def predict(req: PredictRequest):
    return _predict(req.store_id, req.start_date, req.end_date)


@app.get("/predict", response_model=Forecast, summary="Same, via query string")
def predict_get(
    store_id: int = Query(..., examples=[262]),
    start_date: date = Query(..., examples=["2015-08-01"]),
    end_date: date = Query(..., examples=["2015-08-14"]),
):
    return _predict(store_id, start_date, end_date)


@app.get("/health", summary="Liveness plus model provenance")
def health():
    return {
        "status": "ok",
        "model": {
            "type": "LightGBM",
            "config": META["config"],
            "n_features": len(FEATURES),
            "num_boost_round": META["num_boost_round"],
            "target_transform": META["target_transform"],
            "trained_through": LAST_TRAINED,
        },
        "test_metrics": META["test_metrics"],
        "baseline_test_metrics": META["baseline_test_metrics"],
        "servable_range": {"from": str(SERVE_FROM), "to": str(SERVE_TO)},
        "n_stores": len(STORE_IDS),
    }


@app.get("/stores", summary="Store IDs the service can forecast")
def stores(limit: int = 50):
    return {"n_stores": len(STORE_IDS), "sample": STORE_IDS[:limit]}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    """Serve the landing page.

    The page is a separate file with {{TOKEN}} placeholders. The values come
    from the model metadata, so the page cannot show a metric that the served
    model does not have.
    """
    m, b = META["test_metrics"], META["baseline_test_metrics"]
    values = {
        "N_FEATURES": str(len(FEATURES)),
        "N_ROUNDS": str(META["num_boost_round"]),
        "MAPE": f"{m['mape']:.2f}",
        "WAPE": f"{m['wape']:.2f}",
        "RMSE": f"{m['rmse']:,.0f}",
        "BASE_MAPE": f"{b['mape']:.2f}",
        "BASE_WAPE": f"{b['wape']:.2f}",
        "BASE_RMSE": f"{b['rmse']:,.0f}",
        "IMPROVE": f"{(1 - m['mape'] / b['mape']) * 100:.1f}",
        "TRAINED_THROUGH": LAST_TRAINED,
        "SERVE_FROM": str(SERVE_FROM),
        "SERVE_TO": str(SERVE_TO),
        "N_STORES": f"{len(STORE_IDS):,}",
    }
    html = TEMPLATE
    for key, value in values.items():
        html = html.replace("{{" + key + "}}", value)
    return html
