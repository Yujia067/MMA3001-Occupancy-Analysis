"""
Utilities for the MMA3001 Occupancy Analysis Project.

This module contains functions for preparing occupancy
sensor data and evaluating classification results.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def create_features(data):
    """
    Extract time features and encode occupancy status.

    Parameters
    ----------
    data : pandas.DataFrame
        Sensor records containing collecteddate,
        floorspaceid and occupancystatus.

    Returns
    -------
    pandas.DataFrame
        Data containing hour, day_of_week, month
        and the binary target variable.

    Raises
    ------
    KeyError
        If a required column is missing.
    """
    df = data.copy()

    df["collecteddate"] = pd.to_datetime(
        df["collecteddate"], errors="coerce", utc=True
    )

    local_time = df["collecteddate"].dt.tz_convert(
        "Australia/Melbourne"
    )

    df["hour"] = local_time.dt.hour
    df["day_of_week"] = local_time.dt.dayofweek
    df["month"] = local_time.dt.month

    df["target"] = (
        df["occupancystatus"] == "NotOccupied"
    ).astype(int)

    return df


def evaluate_model(y_true, y_pred):
    """
    Calculate binary classification performance metrics.

    Parameters
    ----------
    y_true : array-like
        Actual binary occupancy labels.
    y_pred : array-like
        Predicted binary occupancy labels.

    Returns
    -------
    dict
        Accuracy, balanced accuracy, precision,
        recall and F1 score.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        )
    }
