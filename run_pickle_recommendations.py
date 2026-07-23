import joblib
import pandas as pd

from rf_feature_engineering import MODEL_OUTPUT_PATH, add_engineered_features


RECOMMENDATIONS_OUTPUT_PATH = "recommendations.csv"


def get_sample_inputs() -> pd.DataFrame:
    """Sample input set used to run inference and generate recommendations."""
    return pd.DataFrame(
        [
            {
                "Make": "Toyota",
                "Model": "Camry",
                "Year": 2021,
                "Fuel_Type": "Petrol",
                "Transmission": "Automatic",
                "Engine_Size": 2.0,
                "Mileage": 42000,
                "Horsepower": 165,
                "Torque": 155,
                "Owners": 1,
                "Accident_History": 0,
                "Service_History": "Full Service",
                "Color": "White",
                "Body_Type": "Sedan",
                "Drivetrain": "FWD",
                "Fuel_Efficiency": 36,
                "Location": "CA",
                "Listed_Price": 18500,
            },
            {
                "Make": "BMW",
                "Model": "X5",
                "Year": 2020,
                "Fuel_Type": "Diesel",
                "Transmission": "Automatic",
                "Engine_Size": 3.0,
                "Mileage": 68000,
                "Horsepower": 255,
                "Torque": 300,
                "Owners": 2,
                "Accident_History": 0,
                "Service_History": "Partial Service",
                "Color": "Black",
                "Body_Type": "SUV",
                "Drivetrain": "AWD",
                "Fuel_Efficiency": 28,
                "Location": "TX",
                "Listed_Price": 36500,
            },
            {
                "Make": "Hyundai",
                "Model": "Tucson",
                "Year": 2017,
                "Fuel_Type": "Petrol",
                "Transmission": "Automatic",
                "Engine_Size": 2.0,
                "Mileage": 98000,
                "Horsepower": 175,
                "Torque": 168,
                "Owners": 3,
                "Accident_History": 1,
                "Service_History": "No Service",
                "Color": "Blue",
                "Body_Type": "SUV",
                "Drivetrain": "FWD",
                "Fuel_Efficiency": 31,
                "Location": "FL",
                "Listed_Price": 14200,
            },
        ]
    )


def recommendation_from_prices(predicted_price: float, listed_price: float) -> str:
    ratio = listed_price / predicted_price if predicted_price > 0 else float("inf")

    if ratio <= 0.95:
        return "Strong buy: listed well below estimated value."
    if ratio <= 1.10:
        return "Fair deal: reasonable price, negotiate if possible."
    return "Potentially overpriced: negotiate hard or skip."


def build_risk_note(accident_history: float, owners: float, service_history: str) -> str:
    notes = []
    if pd.notna(accident_history) and accident_history >= 1:
        notes.append("prior accident")
    if pd.notna(owners) and owners >= 3:
        notes.append("multiple owners")
    if isinstance(service_history, str) and service_history.strip().lower() == "no service":
        notes.append("missing service records")

    if not notes:
        return "Low ownership risk profile."
    return "Caution: " + ", ".join(notes) + "."


def main() -> None:
    model = joblib.load(MODEL_OUTPUT_PATH)

    samples = get_sample_inputs()
    listed_prices = samples["Listed_Price"].copy()

    # The training pipeline expects engineered columns, so we apply the same function.
    model_inputs = add_engineered_features(samples.drop(columns=["Listed_Price"]))
    predicted_prices = model.predict(model_inputs)

    results = samples.copy()
    results["Predicted_Price"] = predicted_prices.round(2)
    results["Price_Delta"] = (results["Listed_Price"] - results["Predicted_Price"]).round(2)
    results["Recommendation"] = [
        recommendation_from_prices(pred, listed)
        for pred, listed in zip(predicted_prices, listed_prices)
    ]
    results["Risk_Note"] = [
        build_risk_note(acc, own, svc)
        for acc, own, svc in zip(
            results["Accident_History"],
            results["Owners"],
            results["Service_History"],
        )
    ]

    columns_to_show = [
        "Make",
        "Model",
        "Year",
        "Listed_Price",
        "Predicted_Price",
        "Price_Delta",
        "Recommendation",
        "Risk_Note",
    ]

    print("=== Pickle Inference With Recommendations ===")
    print(results[columns_to_show].to_string(index=False))
    results.to_csv(RECOMMENDATIONS_OUTPUT_PATH, index=False)
    print(f"Saved recommendations: {RECOMMENDATIONS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
