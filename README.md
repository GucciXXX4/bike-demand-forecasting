# 🚲 Bike Demand Forecasting (Machine Learning Project)

## Project Overview

This project builds a Machine Learning model to forecast bike rental demand based on historical usage and environmental conditions. The goal is to help bike rental companies optimize fleet availability, staffing, and infrastructure planning by predicting the number of bikes required at a given time.

The project follows a full ML workflow:
- Data loading
- Cleaning & preprocessing
- Feature engineering
- Model training
- Evaluation
- Prediction
- Model persistence

---

## Problem Statement

Accurately predicting bike demand allows companies to:

- Reduce bike shortages and oversupply
- Improve customer satisfaction
- Optimize operational costs
- Support data-driven business decisions

This project predicts the total number of rentals (`cnt`) using weather, time, and calendar features.

---

## Dataset

Source:  
UCI Bike Sharing Dataset (hourly data)

Features include:

- Season
- Year
- Month
- Hour
- Holiday
- Weekday
- Working day
- Weather situation
- Temperature
- Feels-like temperature
- Humidity
- Wind speed

Target variable:

- `cnt` → Total bike rentals per hour

---

## Project Structure

bike-demand-forecasting/
│
├── bike_demand.py # Main training & prediction script
├── artifacts/
│ └── bike_demand_model.joblib
├── predictions.csv # Generated predictions
├── README.md
├── requirements.txt
├── .gitignore
└── .venv/ # (ignored)

yaml
Αντιγραφή κώδικα

---

## Installation

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
2. Install dependencies
bash
Αντιγραφή κώδικα
pip install -r requirements.txt
Usage
Train the model
bash
Αντιγραφή κώδικα
python bike_demand.py train
This will:

Load the dataset

Train the model

Evaluate performance

Save the trained model to artifacts/

Example output:

yaml
Αντιγραφή κώδικα
MAE :  30.46
RMSE:  48.49
R^2 :  0.926
Run predictions (demo)
bash
Αντιγραφή κώδικα
python bike_demand.py demo
This generates:

Αντιγραφή κώδικα
predictions.csv
containing predicted bike demand.

Model Details
Algorithm: Gradient Boosting Regressor

Feature Scaling: StandardScaler

Evaluation Metrics:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score

The model achieved strong performance with R² ≈ 0.93.

Key Findings
Hour of day and weather conditions strongly affect demand.

Temperature positively correlates with rentals.

Working days show different demand patterns compared to weekends.

Ensemble models significantly improved accuracy over linear models.

Limitations
No external events data (concerts, strikes, etc.)

Only historical weather available

Location-specific (cannot generalize globally)

Future Improvements
Add deep learning (LSTM) for time-series forecasting

Include holiday/event datasets

Deploy as API (FastAPI / Flask)

Real-time predictions

Hyperparameter tuning automation

Technologies Used
Python 3.10

Pandas

NumPy

Scikit-learn

Joblib

Author
Dimitris Gkoutzouvelidis
Machine Learning Project – Bike Demand Forecasting

License
This project is for educational purposes.