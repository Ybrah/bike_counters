from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import radians, sin, cos, sqrt, atan2

def _encode_dates(X):
    """Encode datetime features."""
    X = X.copy()
    
    # Basic time features
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    
    # Additional time features
    X.loc[:, "is_weekend"] = (X["weekday"] >= 5).astype(int)
    X.loc[:, "is_rush_hour"] = ((X["hour"].isin([7,8,9]) | X["hour"].isin([17,18,19]))).astype(int)
    X.loc[:, "is_working_hour"] = ((X["hour"] >= 8) & (X["hour"] <= 18)).astype(int)
    
    # Cyclical encoding
    X.loc[:, "hour_sin"] = np.sin(2 * np.pi * X["hour"]/24)
    X.loc[:, "hour_cos"] = np.cos(2 * np.pi * X["hour"]/24)
    X.loc[:, "month_sin"] = np.sin(2 * np.pi * X["month"]/12)
    X.loc[:, "month_cos"] = np.cos(2 * np.pi * X["month"]/12)
    
    return X.drop(columns=["date"])

def _merge_external_data(X):
    """Merge input data with weather data."""
    X = X.copy()
    
    # Load and process weather data
    file_path = Path(__file__).parent / "data" / "external_data.csv"
    weather_data = pd.read_csv(file_path, parse_dates=["date"])
    
    # Ensure both dataframes use nanosecond datetime
    X["date"] = pd.to_datetime(X["date"]).astype('datetime64[ns]')
    weather_data["date"] = pd.to_datetime(weather_data["date"]).astype('datetime64[ns]')
    
    # Process weather features
    weather_data.loc[:, "temp_celsius"] = weather_data["t"] - 273.15
    weather_data.loc[:, "is_cold"] = (weather_data["temp_celsius"] < 10).astype(int)
    weather_data.loc[:, "is_hot"] = (weather_data["temp_celsius"] > 25).astype(int)
    
    wind_kmh = weather_data["ff"] * 3.6
    weather_data.loc[:, "high_wind"] = (wind_kmh > 20).astype(int)
    
    weather_data.loc[:, "is_raining"] = (weather_data["rr1"] > 0).astype(int)
    weather_data.loc[:, "heavy_rain"] = (weather_data["rr1"] > 5).astype(int)
    
    weather_data.loc[:, "poor_visibility"] = (weather_data["vv"] < 10000).astype(int)
    weather_data.loc[:, "high_humidity"] = (weather_data["u"] > 80).astype(int)
    
    # Keep only necessary columns
    cols_to_keep = ["date", "temp_celsius", "is_cold", "is_hot", "high_wind",
                    "is_raining", "heavy_rain", "poor_visibility", "high_humidity"]
    weather_df = weather_data[cols_to_keep]
    
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    
    # Merge with weather data
    X = pd.merge_asof(
        X.sort_values("date"),
        weather_df.sort_values("date"),
        on="date",
        direction="nearest"
    )
    
    # Sort back to original order and clean up
    X = X.sort_values("orig_index")
    del X["orig_index"]
    
    return X

def _add_location_features(X):
    """Add location-based features."""
    X = X.copy()
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    PARIS_CENTER_LAT = 48.8566
    PARIS_CENTER_LON = 2.3522
    X.loc[:, "distance_to_center"] = X.apply(
        lambda row: haversine_distance(
            row["latitude"], row["longitude"], 
            PARIS_CENTER_LAT, PARIS_CENTER_LON
        ), axis=1
    )
    
    return X

def get_estimator():
    """Create and return the full prediction pipeline."""
    # Get date columns for encoding
    date_cols = ["year", "month", "day", "weekday", "hour", 
                 "is_weekend", "is_rush_hour", "is_working_hour",
                 "hour_sin", "hour_cos", "month_sin", "month_cos"]
    
    # Define categorical columns
    categorical_cols = ["counter_name", "site_name"]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    
    # Create the full pipeline
    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        FunctionTransformer(_add_location_features, validate=False),
        FunctionTransformer(_encode_dates, validate=False),
        preprocessor,
        XGBRegressor()
    )
    
    return pipe

def prepare_and_split_data(train_data, validation_days=30):
    """
    Prepare features and target variables, and split data into train and validation sets.
    
    Parameters
    ----------
    train_data : pd.DataFrame
        Input DataFrame containing the training data
    validation_days : int, optional
        Number of days to use for validation set, by default 30
        
    Returns
    -------
    tuple
        (X_train, X_valid, y_train, y_valid) - Train and validation splits for features and target
    """
    # Prepare features and target
    y = train_data["log_bike_count"].values
    X = train_data.drop(["log_bike_count", "bike_count", "counter_id", "site_id", 
                        "counter_technical_id", "coordinates", "counter_installation_date"], 
                       axis=1)
    
    # Split into train and validation sets
    cutoff_date = X["date"].max() - pd.Timedelta(f"{validation_days} days")
    train_mask = (X["date"] <= cutoff_date)
    X_train, X_valid = X.loc[train_mask], X.loc[~train_mask]
    y_train, y_valid = y[train_mask], y[~train_mask]
    
    return X_train, X_valid, y_train, y_valid

def main():
    """Main function to demonstrate the pipeline."""
    # Load training data
    train_path = Path(__file__).parent / "data" / "train.parquet"
    train_data = pd.read_parquet(train_path)
    
    # Prepare and split data
    X_train, X_valid, y_train, y_valid = prepare_and_split_data(train_data)
    
    # Create and train the pipeline
    pipe = get_estimator()
    pipe.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = pipe.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(f"\nValidation RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()