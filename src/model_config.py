"""Shared constants for training and deployment scoring (must match notebooks)."""

TARGET = "delivery_time_min"

DROP_COLS = [
    "order_id",
    "driver_id",
    "order_date",
    "order_time",
    "pickup_time",
    "restaurant_latitude",
    "restaurant_longitude",
    "delivery_latitude",
    "delivery_longitude",
]
