#!/usr/bin/env python3
"""
Bike demand forecasting (regression) using Bike Sharing hour.csv.

Commands:
  python bike_demand.py train
  python bike_demand.py predict --input path_or_url.csv --output preds.csv

Target: cnt (total rentals per hour)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


DATA_URL = "https://code.datasciencedojo.com/datasciencedojo/datasets/raw/master/Bike%20Sharing/hour.csv"

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "bike_demand_model.joblib"


def load_data(source: str | None = None) -> pd.DataFrame:
    """Load data from URL or local path."""
    src = source or DATA_URL
    return pd.read_csv(src)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Prepare X, y and decide categorical/numeric columns."""
    if "cnt" not in df.columns:
        raise ValueError("Expected target column 'cnt' not found. Are you using hour.csv?")

    # Drop leakage: these sum into cnt
    df = df.drop(columns=[c for c in ["casual", "registered"] if c in df.columns], errors="ignore")

    y = df["cnt"].astype(float)

    # Drop non-useful identifiers
    drop_cols = [c for c in ["instant", "dteday", "cnt"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Categorical-like columns (integers but represent categories)
    cat_cols = [c for c in ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Ensure numeric columns are numeric and fill missing
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())

    # Ensure categorical columns are int and fill missing
    for c in cat_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].mode(dropna=True)[0]).astype(int)

    return X, y, cat_cols, num_cols


def build_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    """Preprocessing + Random Forest model."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    )

    return Pipeline([("prep", preprocessor), ("model", model)])


def train() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print("Loaded rows/cols:", df.shape)
    print("Columns:", list(df.columns))

    X, y, cat_cols, num_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42
    )

    pipe = build_pipeline(cat_cols, num_cols)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)

    print("\nEvaluation (holdout)")
    print("-------------------")
    print(f"MAE :  {mae:.3f}")
    print(f"RMSE:  {rmse:.3f}")
    print(f"R^2 :  {r2:.3f}")

    joblib.dump(
        {
            "model": pipe,
            "cat_cols": cat_cols,
            "num_cols": num_cols
        },
        MODEL_PATH
    )
    print(f"\nModel saved to: {MODEL_PATH}")


def predict(input_path_or_url: str, output_csv: str) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run: python bike_demand.py train")

    bundle = joblib.load(MODEL_PATH)
    model: Pipeline = bundle["model"]

    df = load_data(input_path_or_url)

    # If target exists, drop it (this is "real-world" style)
    df = df.drop(columns=[c for c in ["cnt", "casual", "registered"] if c in df.columns], errors="ignore")

    # Drop identifiers if present
    df = df.drop(columns=[c for c in ["instant", "dteday"] if c in df.columns], errors="ignore")

    preds = model.predict(df)

    out = df.copy()
    out["pred_cnt"] = preds
    out.to_csv(output_csv, index=False)

    print(f"Predictions saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train")

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--input", required=True)
    p_pred.add_argument("--output", default="bike_predictions.csv")

    args = parser.parse_args()

    if args.cmd == "train":
        train()
    elif args.cmd == "predict":
        predict(args.input, args.output)


if __name__ == "__main__":
    main()
