#!/usr/bin/env bash
# End-to-end check against a running service.
#   ./scripts/smoke_test.sh http://localhost:8080
#
# Asserts behaviour, not just liveness: a known date must come back within a
# sane error band, an out-of-range date must be refused, and an unknown store
# must 404. A health check alone would have passed happily while the service
# was returning wrong-store predictions.
set -euo pipefail
BASE="${1:-http://localhost:8080}"
fail() { echo "FAIL: $1"; exit 1; }

echo "--- health"
curl -sf "$BASE/health" > /tmp/h.json || fail "/health unreachable"
python3 -c "
import json;d=json.load(open('/tmp/h.json'))
assert d['status']=='ok', d
print(f\"  model {d['model']['config']}, {d['model']['n_features']} features, \"
      f\"{d['model']['num_boost_round']} rounds\")
print(f\"  test MAPE {d['test_metrics']['mape']:.2f}% vs baseline \"
      f\"{d['baseline_test_metrics']['mape']:.2f}%\")
print(f\"  servable {d['servable_range']['from']} .. {d['servable_range']['to']}\")
"

echo "--- known dates return actuals and a sane error"
curl -sf "$BASE/predict?store_id=262&start_date=2015-07-06&end_date=2015-07-12" > /tmp/p.json \
  || fail "predict on known dates failed"
python3 -c "
import json;d=json.load(open('/tmp/p.json'))
m=d['mape_vs_actual']
assert m is not None, 'actuals missing for a date inside the public dataset'
assert m < 25, f'window MAPE {m} is far outside the expected band'
assert d['n_days']==7, d['n_days']
print(f\"  store 262, 7 days, window MAPE {m:.2f}%\")
"

echo "--- unseen dates predict without actuals"
curl -sf "$BASE/predict?store_id=262&start_date=2015-08-03&end_date=2015-08-09" > /tmp/u.json \
  || fail "predict on unseen dates failed"
python3 -c "
import json;d=json.load(open('/tmp/u.json'))
assert d['mape_vs_actual'] is None, 'claims actuals for dates nobody has seen'
assert all(f['actual_sales'] is None for f in d['forecast'])
assert d['total_predicted_sales'] > 0
print(f\"  7 unseen days, total predicted {d['total_predicted_sales']:,.0f}\")
"

echo "--- out-of-range date is refused"
code=$(curl -s -o /tmp/e.json -w '%{http_code}' \
  "$BASE/predict?store_id=262&start_date=2015-10-01&end_date=2015-10-05")
[ "$code" = "400" ] || fail "expected 400 past the 48-day boundary, got $code"
echo "  400 as expected"

echo "--- unknown store is refused"
code=$(curl -s -o /dev/null -w '%{http_code}' \
  "$BASE/predict?store_id=9999&start_date=2015-07-01&end_date=2015-07-02")
[ "$code" = "404" ] || fail "expected 404 for unknown store, got $code"
echo "  404 as expected"

echo
echo "PASS: all smoke checks"
