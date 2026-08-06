import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = Path(
    os.environ.get(
        "MODEL_FILE",
        str(BASE_DIR / "models" / "replenishment_model.joblib"),
    )
).resolve()

MODEL_ARTIFACT = None
MODEL = None
MODEL_ERROR = ""


def resolve_model(artifact):
    if hasattr(artifact, "predict"):
        return artifact

    if isinstance(artifact, dict):
        selected_name = artifact.get("selected_model")
        models = artifact.get("models")

        if isinstance(selected_name, str) and isinstance(models, dict):
            selected_model = models.get(selected_name)
            if hasattr(selected_model, "predict"):
                return selected_model

        for key in (
            "model",
            "pipeline",
            "regressor",
            "random_forest",
            "random_forest_model",
            "estimator",
        ):
            candidate = artifact.get(key)
            if hasattr(candidate, "predict"):
                return candidate

        for candidate in artifact.values():
            if hasattr(candidate, "predict"):
                return candidate

    raise TypeError("No supported prediction model was found in the model file.")


def load_model():
    global MODEL_ARTIFACT, MODEL, MODEL_ERROR

    try:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

        MODEL_ARTIFACT = joblib.load(MODEL_FILE)
        MODEL = resolve_model(MODEL_ARTIFACT)
        MODEL_ERROR = ""
        return True

    except Exception as exception:
        MODEL_ARTIFACT = None
        MODEL = None
        MODEL_ERROR = str(exception)
        return False


def model_name():
    if MODEL is None:
        return ""

    if hasattr(MODEL, "steps"):
        try:
            return MODEL.steps[-1][1].__class__.__name__
        except Exception:
            pass

    return MODEL.__class__.__name__


def feature_names():
    if isinstance(MODEL_ARTIFACT, dict):
        stored_columns = MODEL_ARTIFACT.get("feature_columns")
        if isinstance(stored_columns, (list, tuple)):
            return [str(column) for column in stored_columns]

    names = getattr(MODEL, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]

    return []


def feature_count():
    value = getattr(MODEL, "n_features_in_", None)
    if value is not None:
        return int(value)

    if hasattr(MODEL, "steps"):
        try:
            value = getattr(MODEL.steps[-1][1], "n_features_in_", None)
            if value is not None:
                return int(value)
        except Exception:
            pass

    return 0


def item_number(item_id):
    matches = re.findall(r"\d+", str(item_id))
    return int(matches[-1]) if matches else 0


def encode_item(item_id):
    if isinstance(MODEL_ARTIFACT, dict):
        for key in (
            "item_encoder",
            "label_encoder",
            "item_id_encoder",
            "encoder",
        ):
            encoder = MODEL_ARTIFACT.get(key)
            if encoder is not None and hasattr(encoder, "transform"):
                try:
                    encoded = encoder.transform([item_id])
                    return float(np.asarray(encoded).reshape(-1)[0])
                except Exception:
                    pass

        for key in ("item_mapping", "item_id_mapping", "item_to_code"):
            mapping = MODEL_ARTIFACT.get(key)
            if isinstance(mapping, dict):
                if item_id in mapping:
                    return float(mapping[item_id])
                if str(item_id) in mapping:
                    return float(mapping[str(item_id)])

    return float(item_number(item_id))


def average(values, size=None):
    selected = values[-size:] if size else values
    return float(np.mean(selected)) if selected else 0.0


def standard_deviation(values, size):
    selected = values[-size:]
    return float(np.std(selected, ddof=1)) if len(selected) > 1 else 0.0


def lag(values, amount):
    return float(values[-amount]) if len(values) >= amount else 0.0


def get_feature_value(name, item_id, current_stock, history, date_value, trend, forecast_days, safety_days):
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    non_zero = [value for value in history if value > 0]

    values = {
        "item_id": item_id,
        "product_id": item_id,
        "sku": item_id,
        "item": item_id,
        "item_code": encode_item(item_id),
        "item_id_encoded": encode_item(item_id),
        "item_encoded": encode_item(item_id),
        "product_code": encode_item(item_id),
        "item_number": item_number(item_id),
        "current_stock": current_stock,
        "stock": current_stock,
        "stock_quantity": current_stock,
        "forecast_days": forecast_days,
        "forecast_horizon": forecast_days,
        "horizon": forecast_days,
        "horizon_days": forecast_days,
        "safety_days": safety_days,
        "day_of_week": date_value.weekday(),
        "weekday": date_value.weekday(),
        "dow": date_value.weekday(),
        "day_of_month": date_value.day,
        "day": date_value.day,
        "month": date_value.month,
        "year": date_value.year,
        "week_of_year": int(date_value.strftime("%V")),
        "week": int(date_value.strftime("%V")),
        "is_weekend": int(date_value.weekday() >= 5),
        "trend_index": trend,
        "trend": trend,
        "time_index": trend,
        "sales_mean": average(history),
        "mean_sales": average(history),
        "average_sales": average(history),
        "daily_average": average(history),
        "average_daily_sales": average(history),
        "non_zero_average": average(non_zero),
        "lag_1": lag(history, 1),
        "lag_2": lag(history, 2),
        "lag_3": lag(history, 3),
        "lag_7": lag(history, 7),
        "lag_14": lag(history, 14),
        "rolling_mean_3": average(history, 3),
        "rolling_mean_7": average(history, 7),
        "rolling_mean_14": average(history, 14),
        "rolling_mean_30": average(history, 30),
        "rolling_avg_3": average(history, 3),
        "rolling_avg_7": average(history, 7),
        "rolling_avg_14": average(history, 14),
        "rolling_avg_30": average(history, 30),
        "rolling_std_7": standard_deviation(history, 7),
        "rolling_std_14": standard_deviation(history, 14),
        "rolling_std_30": standard_deviation(history, 30),
    }

    if key in values:
        return values[key]

    lag_match = re.fullmatch(r"(?:sales_|demand_)?lag_?(\d+)", key)
    if lag_match:
        return lag(history, int(lag_match.group(1)))

    mean_match = re.fullmatch(r"(?:rolling_)?(?:mean|avg|average)_?(\d+)", key)
    if mean_match:
        return average(history, int(mean_match.group(1)))

    return 0.0


def create_input(item_id, current_stock, history, date_value, trend, forecast_days, safety_days):
    names = feature_names()

    if names:
        row = {
            name: get_feature_value(
                name,
                item_id,
                current_stock,
                history,
                date_value,
                trend,
                forecast_days,
                safety_days,
            )
            for name in names
        }
        return pd.DataFrame([row], columns=names)

    count = feature_count()
    if count <= 0:
        raise ValueError("The model does not expose its required input features.")

    candidates = [
        encode_item(item_id),
        lag(history, 1),
        lag(history, 7),
        lag(history, 14),
        average(history, 7),
        average(history, 14),
        standard_deviation(history, 7),
        date_value.weekday(),
        date_value.day,
        date_value.month,
        int(date_value.weekday() >= 5),
        trend,
        current_stock,
        forecast_days,
        safety_days,
        average(history),
    ]

    while len(candidates) < count:
        candidates.append(0.0)

    return np.asarray([candidates[:count]], dtype=float)


def single_prediction(item_id, current_stock, history, date_value, trend, forecast_days, safety_days):
    model_input = create_input(
        item_id,
        current_stock,
        history,
        date_value,
        trend,
        forecast_days,
        safety_days,
    )

    prediction = MODEL.predict(model_input)
    value = float(np.asarray(prediction).reshape(-1)[0])

    if not math.isfinite(value):
        raise ValueError("The model returned an invalid prediction.")

    return max(value, 0.0)


def forecast(item_id, current_stock, daily_sales, forecast_days, safety_days):
    history = [max(float(value), 0.0) for value in daily_sales]

    while len(history) < 14:
        history.insert(0, 0.0)

    predictions = []
    today = datetime.now().date()

    for step in range(1, forecast_days + 1):
        date_value = today + timedelta(days=step)
        predicted_value = single_prediction(
            item_id,
            current_stock,
            history,
            date_value,
            len(history),
            forecast_days,
            safety_days,
        )
        predictions.append(predicted_value)
        history.append(predicted_value)

    predicted_demand = float(np.sum(predictions))
    predicted_daily_average = predicted_demand / forecast_days
    safety_stock = max(int(round(predicted_daily_average * safety_days)), 0)
    recommended_quantity = max(
        int(math.ceil(predicted_demand + safety_stock - current_stock)),
        0,
    )

    return {
        "predicted_daily_average": predicted_daily_average,
        "predicted_demand": predicted_demand,
        "safety_stock": safety_stock,
        "recommended_replenishment_quantity": recommended_quantity,
        "daily_predictions": predictions,
    }


def request_values():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("A valid JSON request body is required.")

    item_id = str(data.get("item_id", "")).strip()
    if not item_id:
        raise ValueError("item_id is required.")

    current_stock = int(data.get("current_stock", 0))
    forecast_days = int(data.get("forecast_days", 7))
    safety_days = float(data.get("safety_days", 2.0))
    daily_sales = data.get("daily_sales")

    if current_stock < 0:
        raise ValueError("current_stock cannot be negative.")
    if forecast_days < 1 or forecast_days > 30:
        raise ValueError("forecast_days must be between 1 and 30.")
    if not math.isfinite(safety_days) or safety_days < 0 or safety_days > 30:
        raise ValueError("safety_days must be between 0 and 30.")
    if not isinstance(daily_sales, list) or len(daily_sales) < 7:
        raise ValueError("daily_sales must contain at least 7 values.")

    clean_sales = []
    for value in daily_sales[-365:]:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("daily_sales contains an invalid value.")
        clean_sales.append(max(number, 0.0))

    return item_id, current_stock, forecast_days, safety_days, clean_sales


@app.get("/")
def index():
    return jsonify(
        service="SmartSME replenishment prediction service",
        status="running",
        model_loaded=MODEL is not None,
    )


@app.get("/health")
def health():
    response = {
        "model_file": str(MODEL_FILE),
        "model_loaded": MODEL is not None,
        "status": "running" if MODEL is not None else "model_unavailable",
    }

    if MODEL_ERROR:
        response["model_error"] = MODEL_ERROR

    return jsonify(response), 200 if MODEL is not None else 503


@app.post("/reload")
def reload_model():
    loaded = load_model()
    response = {
        "success": loaded,
        "model_file": str(MODEL_FILE),
        "model_loaded": loaded,
        "model": model_name() if loaded else "",
    }

    if not loaded:
        response["message"] = MODEL_ERROR

    return jsonify(response), 200 if loaded else 500


@app.post("/predict")
def predict():
    if MODEL is None:
        return jsonify(
            success=False,
            message=MODEL_ERROR or "The prediction model is not loaded.",
        ), 503

    try:
        item_id, current_stock, forecast_days, safety_days, daily_sales = request_values()
        result = forecast(
            item_id,
            current_stock,
            daily_sales,
            forecast_days,
            safety_days,
        )

        return jsonify(
            success=True,
            model=model_name(),
            item_id=item_id,
            current_stock=current_stock,
            forecast_days=forecast_days,
            safety_days=safety_days,
            predicted_daily_average=round(result["predicted_daily_average"], 4),
            predicted_demand=round(result["predicted_demand"], 4),
            safety_stock=result["safety_stock"],
            recommended_replenishment_quantity=result[
                "recommended_replenishment_quantity"
            ],
            daily_predictions=[round(value, 4) for value in result["daily_predictions"]],
        )

    except ValueError as exception:
        return jsonify(success=False, message=str(exception)), 400

    except Exception as exception:
        app.logger.exception("Prediction failed")
        return jsonify(
            success=False,
            message=f"Prediction failed: {exception}",
        ), 500


load_model()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
