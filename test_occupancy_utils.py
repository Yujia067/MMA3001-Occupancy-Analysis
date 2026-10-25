"""Tests for the occupancy analysis utility functions."""

import pandas as pd
from occupancy_utils import create_features, evaluate_model


def test_feature_creation():
    """Check that time features and target labels are correct."""
    data = pd.DataFrame({
        "collecteddate": [
            "2024-06-01 00:00:00+00:00",
            "2024-06-01 01:00:00+00:00"
        ],
        "floorspaceid": ["Room_A", "Room_B"],
        "occupancystatus": ["NotOccupied", "CurrentlyOccupied"]
    })

    result = create_features(data)

    assert result["hour"].tolist() == [10, 11]
    assert result["target"].tolist() == [1, 0]


def test_evaluation():
    """Check classification metric calculations."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = evaluate_model(y_true, y_pred)

    assert result["Accuracy"] == 0.75
    assert result["Recall"] == 1.0
    assert result["Precision"] == 2 / 3
