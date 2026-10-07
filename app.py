"""
app.py -- Uber fare predictor (Task 3, Part C).

Run:  python app.py        then open  http://127.0.0.1:5000

How the Task 2 pipeline is applied to a user's trip
---------------------------------------------------
uber_fare_model.pkl is the WHOLE Task 2 pipeline, not just a regressor:
    FeatureEngineer  (distance, distance to JFK, hour sin/cos, weekend, near-airport, ordinal encodings)
 -> ColumnTransformer (outlier cap on distance + StandardScaler, one-hot for weather)
 -> trained model
So the app only validates the raw form values and hands them over as a one-row DataFrame with the
same raw columns used in training. Nothing is pre-scaled or re-implemented here, which is what
prevents the "model sees raw values it was never trained on" mistake.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# importing this module is also what lets joblib find FeatureEngineer / Winsorizer when unpickling
from uber_pipeline import (RAW_COLUMNS, WEATHER_VALUES, CAR_ORDER, TRAFFIC_ORDER,
                           haversine_km, validate_trip)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'uber_fare_model.pkl')
META_PATH = os.path.join(BASE_DIR, 'model_metadata.json')
MIN_FARE = 2.50                      # NYC minimum fare, same floor used when cleaning the data in Task 1/2

# ---- load the model ONCE when the app starts (never retrained or reloaded per request) ----
try:
    MODEL = joblib.load(MODEL_PATH)
except Exception as exc:             # most common cause: scikit-learn version differs from the one that saved the file
    sys.exit(f'Could not load {MODEL_PATH}\n  -> {exc}\n'
             'If this is a version / pickle error, re-run Task3.ipynb (section 6) in THIS environment '
             'to re-create uber_fare_model.pkl, then start the app again.')
META = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}

app = Flask(__name__)

# Example places so the form is quick to fill (pure UI convenience; the model only sees coordinates)
PLACES = {
    'Times Square':          (40.7580, -73.9855),
    'Central Park (south)':  (40.7648, -73.9730),
    'Grand Central':         (40.7527, -73.9772),
    'Brooklyn Bridge':       (40.7061, -73.9969),
    'Wall Street':           (40.7074, -74.0113),
    'Columbia University':   (40.8075, -73.9626),
    'LaGuardia Airport':     (40.7769, -73.8740),
    'JFK Airport':           (40.6413, -73.7781),
}

DEFAULTS = {
    'pickup_latitude': '40.7580', 'pickup_longitude': '-73.9855',
    'dropoff_latitude': '40.7648', 'dropoff_longitude': '-73.9730',
    'passenger_count': '1', 'pickup_datetime': '2014-06-17T18:30',
    'weather': 'sunny', 'traffic_condition': 'Flow Traffic', 'car_condition': 'Good',
}


def predict_fare(trip):
    """trip = validated raw values. Returns (fare, distance_km). Raises on model failure."""
    row = pd.DataFrame([trip], columns=RAW_COLUMNS)           # one raw row, same columns as training
    fare = float(MODEL.predict(row)[0])
    if not np.isfinite(fare):
        raise ValueError('model returned a non-finite value')
    fare = max(round(fare, 2), MIN_FARE)
    km = float(haversine_km(trip['pickup_latitude'], trip['pickup_longitude'],
                            trip['dropoff_latitude'], trip['dropoff_longitude']))
    return fare, km


def page(form, **extra):
    return render_template('index.html', form=form, places=PLACES, weather_values=WEATHER_VALUES,
                           traffic_values=list(TRAFFIC_ORDER), car_values=list(CAR_ORDER),
                           meta=META, **extra)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return page(DEFAULTS)

    form = request.form.to_dict()
    trip, errors, warnings = validate_trip(form)           # missing / unparsable / out-of-range -> clear messages
    if errors:
        return page(form, errors=errors), 400

    try:
        fare, km = predict_fare(trip)
    except Exception:
        app.logger.exception('prediction failed')
        return page(form, errors=['Something went wrong while computing the fare. '
                                  'Please check the values and try again.']), 500

    result = {'fare': f'{fare:.2f}', 'km': f'{km:.1f}', 'miles': f'{km / 1.609344:.1f}',
              'when': pd.to_datetime(trip['pickup_datetime']).strftime('%a %d %b %Y, %H:%M'),
              'passengers': trip['passenger_count']}
    return page(form, result=result, warnings=warnings)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Same validation + pipeline as the web form, for JSON clients (curl, tests)."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'errors': ['Send a JSON object.']}), 400
    trip, errors, warnings = validate_trip({k: str(v) for k, v in payload.items() if v is not None})
    if errors:
        return jsonify({'errors': errors}), 400
    fare, km = predict_fare(trip)
    return jsonify({'predicted_fare_usd': fare, 'distance_km': round(km, 2), 'warnings': warnings})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
