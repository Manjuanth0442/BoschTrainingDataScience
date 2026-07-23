import pandas as pd
import numpy as np
import joblib
import json
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


RANDOM_STATE = 42
DATA_PATH = "automobile_dataset.csv"
TARGET = "Selling_Price"
MODEL_OUTPUT_PATH = "random_forest_pipeline.pkl"
METRICS_OUTPUT_PATH = "metrics.json"
PARAMS_PATH = "params.yaml"


def load_training_params(path: str = PARAMS_PATH) -> dict:
    """Load train configuration from params.yaml with safe defaults."""
    defaults = {
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
    }

    try:
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except FileNotFoundError:
        return defaults

    train_cfg = config.get("train", {})
    merged = defaults.copy()
    merged.update({k: v for k, v in train_cfg.items() if k in defaults})
    return merged


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional model-friendly features from existing columns."""
    data = df.copy()

    # Price typically depends on depreciation and usage intensity.
    if "Year" in data.columns:
        data["Car_Age"] = 2026 - pd.to_numeric(data["Year"], errors="coerce")

    if "Mileage" in data.columns:
        mileage_numeric = pd.to_numeric(data["Mileage"], errors="coerce")
        data["Mileage_per_Year"] = mileage_numeric / (data["Car_Age"].replace(0, 1))

    if "Horsepower" in data.columns and "Torque" in data.columns:
        hp = pd.to_numeric(data["Horsepower"], errors="coerce")
        tq = pd.to_numeric(data["Torque"], errors="coerce")
        data["Power_Torque_Ratio"] = hp / tq.replace(0, np.nan)

    if "Engine_Size" in data.columns and "Horsepower" in data.columns:
        engine = pd.to_numeric(data["Engine_Size"], errors="coerce")
        hp = pd.to_numeric(data["Horsepower"], errors="coerce")
        data["HP_per_Engine_Size"] = hp / engine.replace(0, np.nan)

    return data


def main() -> None:
    train_params = load_training_params()

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")

    df = add_engineered_features(df)

    X = df.drop(columns=[TARGET])
    y = pd.to_numeric(df[TARGET], errors="coerce")

    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=int(train_params["n_estimators"]),
        max_depth=train_params["max_depth"],
        min_samples_split=int(train_params["min_samples_split"]),
        min_samples_leaf=int(train_params["min_samples_leaf"]),
        random_state=int(train_params["random_state"]),
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("identity", FunctionTransformer(validate=False)),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(train_params["test_size"]),
        random_state=int(train_params["random_state"]),
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    # Persist the full preprocessing + model pipeline for inference reuse.
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("=== Random Forest Regressor Results ===")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"R2 score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Saved model pipeline: {MODEL_OUTPUT_PATH}")

    metrics = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }
    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"Saved metrics: {METRICS_OUTPUT_PATH}")

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    top_n = 12
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    print("\n=== Top Feature Importances ===")
    print(importance_df.head(top_n).to_string(index=False))


if __name__ == "__main__":
    main()
