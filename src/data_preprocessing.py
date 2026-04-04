# =========================
# Imports
# =========================
import pandas as pd
import numpy as np
import os

# =========================
# CONFIG
# =========================
FIG_DIR = "../reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# =========================
# COLUMN RENAMING
# =========================
def rename_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    df.rename(columns={
        "ID": "order_id",
        "Delivery_person_ID": "driver_id",
        "Delivery_person_Age": "driver_age",
        "Delivery_person_Ratings": "driver_rating",
        "Restaurant_latitude": "restaurant_latitude",
        "Restaurant_longitude": "restaurant_longitude",
        "Delivery_location_latitude": "delivery_latitude",
        "Delivery_location_longitude": "delivery_longitude",
        "Order_Date": "order_date",
        "Time_Orderd": "order_time",
        "Time_Order_picked": "pickup_time",
        "Weatherconditions": "weather",
        "Road_traffic_density": "traffic_density",
        "Vehicle_condition": "vehicle_condition",
        "Type_of_order": "order_type",
        "Type_of_vehicle": "vehicle_type",
        "multiple_deliveries": "num_deliveries",
        "Festival": "is_festival",
        "City": "city",
        "Time_taken(min)": "delivery_time_min"
    }, inplace=True)

    return df

# =========================
# DATA CLEANING
# =========================
def clean_data(df):
    df = df.copy()

    # Replace invalid missing values
    df.replace(['NaN', 'NaN ', 'nan', "conditions NaN"], np.nan, inplace=True)

    # Strip whitespace
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    # Fix target (if exists)
    if "delivery_time_min" in df.columns:
        df["delivery_time_min"] = df["delivery_time_min"].astype(str).str.extract(r'(\d+)')
        df["delivery_time_min"] = pd.to_numeric(df["delivery_time_min"], errors='coerce')

    # Fix ratings
    if "driver_rating" in df.columns:
        df["driver_rating"] = pd.to_numeric(df["driver_rating"], errors='coerce')
        df.loc[df["driver_rating"] > 5, "driver_rating"] = np.nan

    # Convert dates
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], format="%d-%m-%Y", errors='coerce')

    if "order_time" in df.columns:
        df["order_time"] = pd.to_datetime(df["order_time"], format="%H:%M:%S", errors='coerce')

    if "pickup_time" in df.columns:
        df["pickup_time"] = pd.to_datetime(df["pickup_time"], format="%H:%M:%S", errors='coerce')

    # Numeric columns
    num_cols = ["driver_age", "driver_rating", "num_deliveries"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    if "num_deliveries" in df.columns:
        df["num_deliveries"] = df["num_deliveries"].astype(int)

    # Categorical columns
    cat_cols = [
        "weather", "traffic_density",
        "order_type", "vehicle_type",
        "is_festival", "city"
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0]).astype("category")

    # Fix weather text
    if "weather" in df.columns:
        df["weather"] = df["weather"].str.replace("conditions ", "", regex=False).str.strip()

    return df

# =========================
# FEATURE ENGINEERING
# =========================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def add_features(df):
    df = df.copy()

    # Distance
    if all(col in df.columns for col in [
        "restaurant_latitude", "restaurant_longitude",
        "delivery_latitude", "delivery_longitude"
    ]):
        df["distance_km"] = haversine(
            df["restaurant_latitude"],
            df["restaurant_longitude"],
            df["delivery_latitude"],
            df["delivery_longitude"]
        )

    # Time features
    if "order_time" in df.columns:
        df["order_hour"] = df["order_time"].dt.hour

    if "order_date" in df.columns:
        df["order_day"] = df["order_date"].dt.day_name()

    if "order_day" in df.columns:
        df["is_weekend"] = df["order_day"].isin(["Saturday", "Sunday"]).astype(int)

    if "order_hour" in df.columns:
        df["is_peak_hour"] = df["order_hour"].isin(
            [12,13,14,15,18,19,20,21,22]
        ).astype(int)

    # Convert categorical
    cat_cols = [
        "weather", "traffic_density",
        "order_type", "vehicle_type",
        "is_festival", "city", "order_day"
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df

# =========================
# OUTLIER HANDLING
# =========================
def handle_outliers(df):
    df = df.copy()

    # Distance outliers
    if "distance_km" in df.columns:
        upper_limit = df["distance_km"].quantile(0.99)
        df = df[df["distance_km"] <= upper_limit]

    # Driver age validation
    if "driver_age" in df.columns:
        df = df[(df["driver_age"] >= 18) & (df["driver_age"] <= 40)]

    return df

# =========================
# FULL PIPELINE
# =========================
def preprocess_pipeline(df):
    df = rename_columns(df)
    df = clean_data(df)
    df = add_features(df)
    df = handle_outliers(df)
    return df


def preprocess_for_scoring(df):
    """Same as training prep but does not drop rows (no handle_outliers). Use for deployment."""
    df = rename_columns(df)
    df = clean_data(df)
    df = add_features(df)
    return df

# =========================
# LOAD CLEAN DATA
# =========================
def load_clean_data(path):
    import pandas as pd
    
    df = pd.read_csv(path)

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["order_time"] = pd.to_datetime(df["order_time"], errors="coerce")
    df["pickup_time"] = pd.to_datetime(df["pickup_time"], errors="coerce")

    cat_cols = [
        "weather", "traffic_density",
        "order_type", "vehicle_type",
        "is_festival", "city", "order_day"
    ]

    for col in cat_cols:
        df[col] = df[col].astype("category")

    df["order_hour"] = df["order_hour"].astype("Int64")

    return df