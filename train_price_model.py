#!/usr/bin/env python3
"""Train and run an automobile selling price prediction model.

Usage:
  python train_price_model.py --csv automobile_dataset.csv --train
  python train_price_model.py --predict
    python train_price_model.py --recommendations
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COL = "Selling_Price"
MODEL_PATH = Path("automobile_price_model.joblib")
PKL_MODEL_PATH = Path("automobile_price_model.pkl")
META_PATH = Path("automobile_price_model_meta.json")
CURRENT_YEAR = datetime.now().year


@dataclass
class TrainingArtifacts:
    model: Pipeline
    raw_feature_columns: List[str]
    numeric_ranges: Dict[str, Dict[str, float]]
    categorical_values: Dict[str, List[str]]


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {csv_path}")

    numeric_hint_cols = [
        "Year",
        "Engine_Size",
        "Mileage",
        "Horsepower",
        "Torque",
        "Owners",
        "Accident_History",
        "Fuel_Efficiency",
        TARGET_COL,
    ]

    for col in numeric_hint_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET_COL]).copy()
    if df.empty:
        raise ValueError("No rows left after dropping missing target values.")

    return df


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready engineered features from raw automobile attributes."""
    X = X.copy()

    if "Year" in X.columns:
        year = pd.to_numeric(X["Year"], errors="coerce")
        vehicle_age = (CURRENT_YEAR - year).clip(lower=0)
        X["Vehicle_Age"] = vehicle_age

    if "Mileage" in X.columns and "Vehicle_Age" in X.columns:
        mileage = pd.to_numeric(X["Mileage"], errors="coerce")
        age_nonzero = X["Vehicle_Age"].replace(0, 1)
        X["Mileage_Per_Year"] = mileage / age_nonzero

    if "Horsepower" in X.columns and "Engine_Size" in X.columns:
        hp = pd.to_numeric(X["Horsepower"], errors="coerce")
        engine_size = pd.to_numeric(X["Engine_Size"], errors="coerce")
        X["Horsepower_Per_Liter"] = hp / engine_size.replace(0, np.nan)

    if "Torque" in X.columns and "Engine_Size" in X.columns:
        torque = pd.to_numeric(X["Torque"], errors="coerce")
        engine_size = pd.to_numeric(X["Engine_Size"], errors="coerce")
        X["Torque_Per_Liter"] = torque / engine_size.replace(0, np.nan)

    if "Fuel_Efficiency" in X.columns and "Engine_Size" in X.columns:
        mpg = pd.to_numeric(X["Fuel_Efficiency"], errors="coerce")
        engine_size = pd.to_numeric(X["Engine_Size"], errors="coerce")
        X["Efficiency_Per_Liter"] = mpg / engine_size.replace(0, np.nan)

    if "Transmission" in X.columns:
        tx = X["Transmission"].astype(str).str.lower()
        X["Is_Automatic"] = tx.str.contains("automatic", na=False).astype(float)

    if "Service_History" in X.columns:
        service_map = {
            "No Service": 0.0,
            "Partial Service": 1.0,
            "Full Service": 2.0,
        }
        X["Service_History_Score"] = X["Service_History"].map(service_map)

    if "Make" in X.columns:
        luxury_brands = {"BMW", "Audi", "Mercedes-Benz"}
        X["Is_Luxury_Brand"] = X["Make"].isin(luxury_brands).astype(float)

    return X


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=700,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def build_metadata(X: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[str]]]:
    numeric_ranges: Dict[str, Dict[str, float]] = {}
    categorical_values: Dict[str, List[str]] = {}

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            col_non_na = X[col].dropna()
            if not col_non_na.empty:
                numeric_ranges[col] = {
                    "min": float(col_non_na.min()),
                    "max": float(col_non_na.max()),
                    "median": float(col_non_na.median()),
                }
        else:
            values = sorted(X[col].dropna().astype(str).unique().tolist())
            categorical_values[col] = values

    return numeric_ranges, categorical_values


def train(csv_path: Path) -> None:
    df = load_and_prepare_data(csv_path)
    X_raw = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )

    X_train = add_engineered_features(X_raw_train)
    X_test = add_engineered_features(X_raw_test)

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    numeric_ranges, categorical_values = build_metadata(X_raw)
    artifacts = TrainingArtifacts(
        model=pipeline,
        raw_feature_columns=X_raw.columns.tolist(),
        numeric_ranges=numeric_ranges,
        categorical_values=categorical_values,
    )

    joblib.dump(artifacts.model, MODEL_PATH)
    with PKL_MODEL_PATH.open("wb") as pkl_file:
        pickle.dump(artifacts.model, pkl_file)
    META_PATH.write_text(
        json.dumps(
            {
                "raw_feature_columns": artifacts.raw_feature_columns,
                "numeric_ranges": artifacts.numeric_ranges,
                "categorical_values": artifacts.categorical_values,
            },
            indent=2,
        )
    )

    print("Training complete.")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved model to: {PKL_MODEL_PATH}")
    print(f"Saved metadata to: {META_PATH}")
    print("Evaluation on holdout set:")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  R2   : {r2:.4f}")


def _parse_numeric_input(
    col: str, prompt: str, ranges: Dict[str, Dict[str, float]]
) -> float | None:
    info = ranges.get(col)
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a valid number, or press Enter to skip.")
            continue

        if info:
            low, high = info["min"], info["max"]
            if value < low or value > high:
                print(
                    f"Note: typical {col} range in training data is [{low:.2f}, {high:.2f}]."
                )
        return value


def _parse_categorical_input(
    col: str, prompt: str, allowed_values: Dict[str, List[str]]
) -> str | None:
    allowed = allowed_values.get(col, [])
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        if not allowed:
            return raw

        lookup = {v.lower(): v for v in allowed}
        if raw.lower() in lookup:
            return lookup[raw.lower()]

        sample = ", ".join(allowed[:10])
        print(f"Unknown value for {col}. Example valid values: {sample}")


def predict_interactive() -> None:
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Train first with --train and --csv."
        )

    model: Pipeline = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())

    feature_columns: List[str] = meta.get("raw_feature_columns", meta.get("feature_columns", []))
    numeric_ranges: Dict[str, Dict[str, float]] = meta["numeric_ranges"]
    categorical_values: Dict[str, List[str]] = meta["categorical_values"]

    print("Enter car details. Press Enter to skip any field.")
    row: Dict[str, object] = {}

    for col in feature_columns:
        if col in numeric_ranges:
            med = numeric_ranges[col].get("median")
            prompt = f"{col} (numeric, median {med:.2f}): " if med is not None else f"{col} (numeric): "
            row[col] = _parse_numeric_input(col, prompt, numeric_ranges)
        else:
            examples = categorical_values.get(col, [])[:6]
            ex_text = f" e.g. {', '.join(examples)}" if examples else ""
            prompt = f"{col} (text,{ex_text}): " if ex_text else f"{col} (text): "
            row[col] = _parse_categorical_input(col, prompt, categorical_values)

    input_df_raw = pd.DataFrame([row], columns=feature_columns)
    input_df = add_engineered_features(input_df_raw)
    pred = model.predict(input_df)[0]
    print(f"\nPredicted selling price: ${pred:,.2f}")


def run_recommendations() -> None:
    if not PKL_MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Pickle model or metadata not found. Train first with --train and --csv."
        )

    with PKL_MODEL_PATH.open("rb") as pkl_file:
        model: Pipeline = pickle.load(pkl_file)
    meta = json.loads(META_PATH.read_text())
    feature_columns: List[str] = meta.get("raw_feature_columns", meta.get("feature_columns", []))

    # A sensible mix of economical, family, and premium configurations.
    candidate_inputs: List[Dict[str, object]] = [
        {
            "Make": "Toyota",
            "Model": "Camry",
            "Year": 2022,
            "Fuel_Type": "Petrol",
            "Transmission": "Automatic",
            "Engine_Size": 2.0,
            "Mileage": 35000,
            "Horsepower": 180,
            "Torque": 175,
            "Owners": 1,
            "Accident_History": 0,
            "Service_History": "Full Service",
            "Color": "Gray",
            "Body_Type": "Sedan",
            "Drivetrain": "FWD",
            "Fuel_Efficiency": 35,
            "Location": "CA",
        },
        {
            "Make": "Honda",
            "Model": "CR-V",
            "Year": 2021,
            "Fuel_Type": "Hybrid",
            "Transmission": "Automatic",
            "Engine_Size": 2.4,
            "Mileage": 30000,
            "Horsepower": 190,
            "Torque": 180,
            "Owners": 1,
            "Accident_History": 0,
            "Service_History": "Full Service",
            "Color": "Blue",
            "Body_Type": "SUV",
            "Drivetrain": "AWD",
            "Fuel_Efficiency": 40,
            "Location": "TX",
        },
        {
            "Make": "Ford",
            "Model": "F-150",
            "Year": 2020,
            "Fuel_Type": "Petrol",
            "Transmission": "Automatic",
            "Engine_Size": 3.5,
            "Mileage": 45000,
            "Horsepower": 320,
            "Torque": 310,
            "Owners": 1,
            "Accident_History": 0,
            "Service_History": "Partial Service",
            "Color": "Black",
            "Body_Type": "Truck",
            "Drivetrain": "RWD",
            "Fuel_Efficiency": 20,
            "Location": "FL",
        },
        {
            "Make": "BMW",
            "Model": "X3",
            "Year": 2023,
            "Fuel_Type": "Petrol",
            "Transmission": "Automatic",
            "Engine_Size": 2.0,
            "Mileage": 12000,
            "Horsepower": 245,
            "Torque": 250,
            "Owners": 1,
            "Accident_History": 0,
            "Service_History": "Full Service",
            "Color": "White",
            "Body_Type": "SUV",
            "Drivetrain": "AWD",
            "Fuel_Efficiency": 28,
            "Location": "NY",
        },
        {
            "Make": "Hyundai",
            "Model": "Sonata",
            "Year": 2019,
            "Fuel_Type": "Petrol",
            "Transmission": "Automatic",
            "Engine_Size": 2.0,
            "Mileage": 60000,
            "Horsepower": 170,
            "Torque": 160,
            "Owners": 2,
            "Accident_History": 0,
            "Service_History": "Partial Service",
            "Color": "Silver",
            "Body_Type": "Sedan",
            "Drivetrain": "FWD",
            "Fuel_Efficiency": 33,
            "Location": "IL",
        },
    ]

    candidates_df = pd.DataFrame(candidate_inputs)
    candidates_df = candidates_df.reindex(columns=feature_columns)
    candidates_df_engineered = add_engineered_features(candidates_df)
    predictions = model.predict(candidates_df_engineered)

    result_df = candidates_df.copy()
    result_df["Predicted_Selling_Price"] = predictions
    result_df = result_df.sort_values("Predicted_Selling_Price", ascending=False).reset_index(drop=True)

    print("\nPredicted prices for recommended input set (highest to lowest):")
    for idx, row in result_df.iterrows():
        print(
            f"{idx + 1}. {row['Year']} {row['Make']} {row['Model']} "
            f"({row['Body_Type']}, {row['Fuel_Type']}) -> ${row['Predicted_Selling_Price']:,.2f}"
        )

    best = result_df.iloc[0]
    print("\nTop recommendation:")
    print(
        f"{best['Year']} {best['Make']} {best['Model']} with predicted price "
        f"${best['Predicted_Selling_Price']:,.2f}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automobile price prediction model")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("automobile_dataset.csv"),
        help="Path to the CSV dataset",
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--predict", action="store_true", help="Run interactive prediction"
    )
    parser.add_argument(
        "--recommendations",
        action="store_true",
        help="Run pickle model on a sensible set of inputs and show ranked recommendations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train and not args.predict and not args.recommendations:
        raise SystemExit("Choose at least one action: --train, --predict, and/or --recommendations")

    if args.train:
        train(args.csv)

    if args.predict:
        predict_interactive()

    if args.recommendations:
        run_recommendations()


if __name__ == "__main__":
    main()
