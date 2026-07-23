from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from rf_feature_engineering import MODEL_OUTPUT_PATH, add_engineered_features


class CarInput(BaseModel):
    Make: Optional[str] = None
    Model: Optional[str] = None
    Year: Optional[int] = None
    Fuel_Type: Optional[str] = None
    Transmission: Optional[str] = None
    Engine_Size: Optional[float] = None
    Mileage: Optional[float] = None
    Horsepower: Optional[float] = None
    Torque: Optional[float] = None
    Owners: Optional[float] = None
    Accident_History: Optional[float] = None
    Service_History: Optional[str] = None
    Color: Optional[str] = None
    Body_Type: Optional[str] = None
    Drivetrain: Optional[str] = None
    Fuel_Efficiency: Optional[float] = None
    Location: Optional[str] = None
    Listed_Price: Optional[float] = Field(
        default=None,
        description="Optional listing price used for deal recommendation",
    )


class PredictionResponse(BaseModel):
    predicted_price: float
    listed_price: Optional[float] = None
    price_delta: Optional[float] = None
    recommendation: Optional[str] = None


app = FastAPI(
    title="Automobile Price Prediction API",
    description="Random Forest model served with FastAPI for automobile price prediction.",
    version="1.0.0",
)

model = joblib.load(MODEL_OUTPUT_PATH)


def recommendation_from_prices(predicted_price: float, listed_price: float) -> str:
    ratio = listed_price / predicted_price if predicted_price > 0 else float("inf")

    if ratio <= 0.95:
        return "Strong buy: listed well below estimated value."
    if ratio <= 1.10:
        return "Fair deal: reasonable price, negotiate if possible."
    return "Potentially overpriced: negotiate hard or skip."


def to_prediction_response(car: CarInput, predicted_price: float) -> PredictionResponse:
    listed_price = car.Listed_Price

    if listed_price is None:
        return PredictionResponse(predicted_price=round(predicted_price, 2))

    price_delta = round(listed_price - predicted_price, 2)
    recommendation = recommendation_from_prices(predicted_price, listed_price)

    return PredictionResponse(
        predicted_price=round(predicted_price, 2),
        listed_price=round(listed_price, 2),
        price_delta=price_delta,
        recommendation=recommendation,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(car: CarInput) -> PredictionResponse:
    raw = car.model_dump()
    feature_row = {key: value for key, value in raw.items() if key != "Listed_Price"}

    input_df = pd.DataFrame([feature_row])
    model_input = add_engineered_features(input_df)
    predicted_price = float(model.predict(model_input)[0])

    return to_prediction_response(car, predicted_price)


@app.post("/predict-batch", response_model=list[PredictionResponse])
def predict_batch(cars: list[CarInput]) -> list[PredictionResponse]:
    rows = []
    for car in cars:
        raw = car.model_dump()
        feature_row = {key: value for key, value in raw.items() if key != "Listed_Price"}
        rows.append(feature_row)

    input_df = pd.DataFrame(rows)
    model_input = add_engineered_features(input_df)
    predictions = model.predict(model_input)

    return [
        to_prediction_response(car, float(predicted_price))
        for car, predicted_price in zip(cars, predictions)
    ]
