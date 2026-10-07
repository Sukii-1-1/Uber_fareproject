
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Constants carried over from Task 1 / Task 2

JFK = (40.6413, -73.7781)          # (lat, lon)
LGA = (40.7769, -73.8740)

CAR_ORDER = {'Bad': 0, 'Good': 1, 'Very Good': 2, 'Excellent': 3}
TRAFFIC_ORDER = {'Flow Traffic': 0, 'Dense Traffic': 1, 'Congested Traffic': 2}
WEATHER_VALUES = ['cloudy', 'rainy', 'stormy', 'sunny', 'windy']

# the raw columns a trip must contain (what the web form collects)
RAW_COLUMNS = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
               'passenger_count', 'pickup_datetime', 'weather', 'traffic_condition', 'car_condition']

# the model-ready columns, in the same order as Task 2
NUMERIC_COLS = ['distance', 'jfk_km', 'passenger_count', 'year', 'hour_sin', 'hour_cos',
                'car_condition_ord', 'traffic_condition_ord']
CAT_COLS = ['weather']
PASSTHROUGH_COLS = ['is_weekend', 'is_near_airport']

# Same cleaning limits as Task 1 / Task 2 (NYC bounding box, passengers 1-6, distance >= 50 m)
LAT_RANGE = (40.5, 41.0)
LON_RANGE = (-74.3, -73.7)
MIN_DISTANCE_KM = 0.05


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (identical formula to Task 1 / Task 2)."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))



class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Raw trip columns -> the engineered columns the model was trained on.

    Stateless (nothing is learned from data), so it can never leak information from the test set.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        when = pd.to_datetime(X['pickup_datetime'])

        out = pd.DataFrame(index=X.index)
        out['distance'] = haversine_km(X['pickup_latitude'], X['pickup_longitude'],
                                       X['dropoff_latitude'], X['dropoff_longitude'])
        out['jfk_km'] = haversine_km(X['pickup_latitude'], X['pickup_longitude'], *JFK)
        lga_km = haversine_km(X['pickup_latitude'], X['pickup_longitude'], *LGA)
        out['passenger_count'] = X['passenger_count'].astype(float)
        out['year'] = when.dt.year.astype(float)
        out['hour_sin'] = np.sin(2 * np.pi * when.dt.hour / 24)
        out['hour_cos'] = np.cos(2 * np.pi * when.dt.hour / 24)
        out['car_condition_ord'] = X['car_condition'].map(CAR_ORDER).astype(float)
        out['traffic_condition_ord'] = X['traffic_condition'].map(TRAFFIC_ORDER).astype(float)
        out['is_weekend'] = (when.dt.dayofweek >= 5).astype(int)
        out['is_near_airport'] = ((out['jfk_km'] < 2) | (lga_km < 2)).astype(int)
        out['weather'] = X['weather']
        return out



class Winsorizer(BaseEstimator, TransformerMixin):
    """Caps the 'distance' column (always column 0 of NUMERIC_COLS) at a quantile learned from training data."""

    def __init__(self, upper_q=0.99):
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.cap_ = np.quantile(X[:, 0], self.upper_q)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        X[:, 0] = np.minimum(X[:, 0], self.cap_)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor():
    """Encoding + outlier capping + scaling, exactly as in Task 2 (section 8)."""
    numeric_pipeline = Pipeline([('cap_outliers', Winsorizer(upper_q=0.99)),
                                 ('scale', StandardScaler())])
    return ColumnTransformer([
        ('num', numeric_pipeline, NUMERIC_COLS),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CAT_COLS),
        ('pass', 'passthrough', PASSTHROUGH_COLS),
    ])


def build_pipeline(model):
    """Full raw-trip -> fare pipeline: FeatureEngineer -> preprocessor -> model."""
    return Pipeline([('features', FeatureEngineer()),
                     ('preprocess', build_preprocessor()),
                     ('model', model)])



def validate_trip(form):
    """Check raw form values. Returns (trip_dict_or_None, errors_list, warnings_list)."""
    errors, warnings = [], []
    trip = {}

    def num(field, label, lo=None, hi=None, integer=False):
        raw = (form.get(field) or '').strip()
        if raw == '':
            errors.append(f'{label} is required.')
            return None
        try:
            value = float(raw)
        except ValueError:
            errors.append(f'{label} must be a number (you entered "{raw}").')
            return None
        if not np.isfinite(value):
            errors.append(f'{label} must be a finite number.')
            return None
        if integer and value != int(value):
            errors.append(f'{label} must be a whole number.')
            return None
        if lo is not None and not (lo <= value <= hi):
            errors.append(f'{label} must be between {lo} and {hi} (you entered {value:g}).')
            return None
        return int(value) if integer else value

    trip['pickup_latitude'] = num('pickup_latitude', 'Pickup latitude', *LAT_RANGE)
    trip['pickup_longitude'] = num('pickup_longitude', 'Pickup longitude', *LON_RANGE)
    trip['dropoff_latitude'] = num('dropoff_latitude', 'Dropoff latitude', *LAT_RANGE)
    trip['dropoff_longitude'] = num('dropoff_longitude', 'Dropoff longitude', *LON_RANGE)
    trip['passenger_count'] = num('passenger_count', 'Passenger count', 1, 6, integer=True)

    # date / time
    raw_dt = (form.get('pickup_datetime') or '').strip()
    if raw_dt == '':
        errors.append('Pickup date and time is required.')
    else:
        try:
            when = pd.to_datetime(raw_dt)
            if pd.isna(when):
                raise ValueError
            if when.year < 2009:
                errors.append('Pickup date must be 2009 or later (the model was trained on 2009-2015 trips).')
            elif when.year > 2030:
                errors.append('Pickup date must be before 2031.')
            else:
                trip['pickup_datetime'] = when.strftime('%Y-%m-%d %H:%M:%S')
                if when.year > 2015:
                    warnings.append('The model was trained on 2009-2015 trips, so a fare for a later year '
                                    'is an extrapolation (it uses 2015-level prices).')
        except (ValueError, TypeError):
            errors.append(f'Pickup date and time could not be read (you entered "{raw_dt}").')

    # categorical fields: must be one of the known values
    for field, label, allowed in [('weather', 'Weather', WEATHER_VALUES),
                                  ('traffic_condition', 'Traffic condition', list(TRAFFIC_ORDER)),
                                  ('car_condition', 'Car condition', list(CAR_ORDER))]:
        value = (form.get(field) or '').strip()
        if value == '':
            errors.append(f'{label} is required.')
        elif value not in allowed:
            errors.append(f'{label} "{value}" is not a valid option.')
        else:
            trip[field] = value

    # trip-level rule from Task 1/2 cleaning: a real trip is at least 50 m long
    coords = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    if all(trip.get(c) is not None for c in coords):
        d = float(haversine_km(trip['pickup_latitude'], trip['pickup_longitude'],
                               trip['dropoff_latitude'], trip['dropoff_longitude']))
        if d < MIN_DISTANCE_KM:
            errors.append('Pickup and dropoff are the same place (under 50 m apart) - that is not a real trip.')

    if errors:
        return None, errors, warnings
    return trip, errors, warnings
