"""Calculate the error measures.

The rule for closed days: predict exactly 0, then remove them from each
metric. A closed store sells nothing, so this is not a forecast. Percentage
error also divides by the actual value, which is not possible at zero. Closed
days are 17% of the rows, so the rule changes the numbers a lot.
"""

import numpy as np
import pandas as pd


def _open_mask(y_true, is_open):
    """Rows the model is actually judged on."""
    if is_open is None:
        return np.asarray(y_true) > 0
    return np.asarray(is_open).astype(bool)


def rmse(y_true, y_pred, is_open=None):
    """Return the root mean squared error, in euros.

    The measure squares each error before it takes the mean. Large errors
    therefore control the result. It answers the question: how many euros is
    the forecast wrong by? Store sizes differ by a factor of five, so the
    largest stores control this number.
    """
    m = _open_mask(y_true, is_open)
    err = np.asarray(y_true)[m] - np.asarray(y_pred)[m]
    return float(np.sqrt(np.mean(err ** 2)))


def mape(y_true, y_pred, is_open=None):
    """Return the mean absolute percentage error, in percent.

    The measure divides each error by the actual sales for that day. A small
    store and a large store therefore count the same.

    The measure has a known limit. It is not symmetric. A forecast that is too
    high costs more than a forecast that is too low by the same amount. A
    model tuned only on MAPE will predict low.
    """
    m = _open_mask(y_true, is_open)
    yt = np.asarray(y_true, dtype=float)[m]
    yp = np.asarray(y_pred, dtype=float)[m]
    m2 = yt > 0                      # guard the 54 open-but-zero-sales rows
    return float(np.mean(np.abs(yt[m2] - yp[m2]) / yt[m2]) * 100)


def wape(y_true, y_pred, is_open=None):
    """Return the weighted absolute percentage error, in percent.

    The measure divides the total absolute error by the total actual sales. It
    works when single values are zero, and it is symmetric. Retail teams use
    it for these reasons. Read it as: the forecast is wrong by X percent of
    total revenue.
    """
    m = _open_mask(y_true, is_open)
    yt = np.asarray(y_true, dtype=float)[m]
    yp = np.asarray(y_pred, dtype=float)[m]
    return float(np.sum(np.abs(yt - yp)) / np.sum(yt) * 100)


def evaluate(y_true, y_pred, is_open=None, label=""):
    """All three measures as a dict, plus the row count they were computed on."""
    m = _open_mask(y_true, is_open)
    return {
        "label": label,
        "n_scored": int(m.sum()),
        "rmse": rmse(y_true, y_pred, is_open),
        "mape": mape(y_true, y_pred, is_open),
        "wape": wape(y_true, y_pred, is_open),
    }


def results_table(rows):
    """Format a list of evaluate() dicts for printing."""
    df = pd.DataFrame(rows).set_index("label")
    return df.round({"rmse": 1, "mape": 2, "wape": 2}).to_string()
