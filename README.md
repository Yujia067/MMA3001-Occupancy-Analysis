# MMA3001 - Room Occupancy Analysis

## Project Overview
This project investigates whether time-based sensor data
can be used to predict room occupancy status.

## Dataset
Monash Smart Infrastructure Occupancy Sensor Dataset.

The original dataset contains 9,091,732 records.
Every 30th record was selected for the machine learning
experiment, resulting in 303,058 sampled records.

The original sensor dataset is not included in this repository.

## Research Aim
To analyse room occupancy record patterns and evaluate
the performance of Logistic Regression for predicting
the NotOccupied status.

## Methods
- Data cleaning and feature engineering
- Exploratory data analysis
- Logistic Regression
- Majority-class baseline comparison
- Decision threshold optimisation
- Verification and computational performance testing

## Input Features
- Hour of day
- Day of week
- Month
- Floor space ID

## Target
1 = NotOccupied
0 = Other occupancy status

## Results
The selected decision threshold was 0.70.

The optimised model achieved:
- Accuracy: 0.9405
- Balanced Accuracy: 0.5656
- Precision: 0.1481
- Recall: 0.1638
- F1 Score: 0.1555

The model showed limited ability to identify the minority
class. Threshold optimisation improved F1 but reduced recall.

## Reproducibility
Open MMA3001_Occupancy_Analysis2.ipynb in Google Colab.
Upload the original occupancy dataset ZIP to /content/.
Run the notebook cells in order.

## Documentation
HTML documentation is available in the docs folder.

## Testing
Unit tests are provided in test_occupancy_utils.py.
Test results are recorded in test_report.txt.

## Limitations
The dataset is highly imbalanced.
The analysis predicts sensor record status, not the actual
duration of room occupancy.
The sampling method may introduce bias.

## AI Use
AI tools were used to assist with code development,
debugging, documentation and explanations.
All AI-generated content must be reviewed and verified
before final submission.
