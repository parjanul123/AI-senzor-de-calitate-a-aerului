from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Literal, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from app.models.train_model import (
    train_and_save_isolation_forest,
    train_and_save_random_forest,
    train_and_save_svm,
)
from app.models.xgboost_model import train_and_save_xgboost
from app.core.database import get_measurements
from app.services.chatbot import get_chatbot_reply
from app.services.anomaly_detector import detect_anomaly as detect_anomaly_service
from app.services.predictor import build_forecast, predict_air_quality, summarize_forecast_average

app = FastAPI(
    title="Air Quality AI API",
    version="1.0.0",
    description="Backend pentru monitorizarea calității aerului și integrarea cu modele AI"
)

# Enable CORS for Streamlit and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.latest_prediction = None
app.state.latest_training = None

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    message: str
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    model_type: Optional[str] = None
    algorithm_comparison: list[dict[str, Any]] | None = None
    input_values: dict[str, float] | None = None
    feature_assessment: dict[str, Any] | None = None
    source_measurement: dict[str, Any] | None = None
    forecast: list[dict[str, Any]] | None = None
    forecast_average: dict[str, Any] | None = None


def _parse_forecast_horizons(raw_horizons: str) -> list[int]:
    parsed: list[int] = []
    for token in raw_horizons.split(","):
        cleaned = token.strip()
        if not cleaned:
            continue
        try:
            value = int(cleaned)
        except ValueError as exc:
            raise ValueError(f"Valoarea '{cleaned}' din forecast_horizons nu este un numar intreg valid.") from exc
        if value <= 0:
            raise ValueError("Toate valorile din forecast_horizons trebuie sa fie > 0.")
        parsed.append(value)

    if not parsed:
        raise ValueError("forecast_horizons trebuie sa contina cel putin un orizont orar valid.")

    return sorted(set(parsed))


def _resolve_forecast_lookback_hours(horizons: list[int], explicit_lookback: int | None) -> int:
    if explicit_lookback is not None:
        return int(explicit_lookback)

    max_horizon = max(horizons)
    # Automatically choose enough history for trend stability while respecting limits.
    return max(24, min(720, max_horizon * 4))


class TrainRequest(BaseModel):
    dataset_name: Optional[str] = None
    notes: Optional[str] = None
    training_model: str = "random_forest"
    aggregation_hours: int = Field(default=24, ge=1, le=720)
    aggregation_minutes: Optional[int] = Field(default=None, ge=1, le=60)
    allow_derived_label_fallback: bool = Field(
        default=False,
        description=(
            "Permite fallback la etichete derivate din feature-uri cand lipsesc etichete independente. "
            "Activeaza doar pentru demo, deoarece introduce data leakage."
        ),
    )


class AnomalyRequest(BaseModel):
    sensor_id: Optional[str] = None
    temperature: float
    humidity: float
    pressure: Optional[float] = None
    co2: float
    pm1: Optional[float] = None
    pm25: float
    pm10: float
    voc: Optional[float] = None
    light: Optional[float] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "air-quality-ai-api"}


@app.get("/health/data")
def data_health_check():
    try:
        measurements = get_measurements(limit=1, descending=True, raise_on_error=True)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if measurements.empty:
        raise HTTPException(
            status_code=503,
            detail="Tabela 'measurements' nu conține date în proiectul Supabase configurat pentru Railway.",
        )

    return {"status": "ok", "service": "supabase-measurements", "rows_available": True}


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict", response_model=PredictionResponse)
def predict(
    model_type: Literal["random_forest", "xgboost", "svm"] = Query(default="random_forest"),
    compare_models: bool = Query(default=False),
    use_hourly_average: bool = Query(default=False),
    aggregation_hours: int = Query(default=1, ge=1, le=168),
    include_forecast: bool = Query(default=False),
    forecast_horizons: str = Query(default="1,3,6,12,24"),
    forecast_lookback_hours: int | None = Query(default=None, ge=2, le=720),
):
    try:
        prediction, confidence, feature_values, feature_assessment = predict_air_quality(
            use_hourly_average=use_hourly_average,
            aggregation_hours=aggregation_hours,
            model_type=model_type,
        )
    except RuntimeError as exc:
        error_msg = str(exc)
        raise HTTPException(status_code=400, detail=error_msg) from exc
    except Exception as exc:
        error_msg = f"Eroare internă la predicție: {str(exc)}"
        raise HTTPException(status_code=500, detail=error_msg) from exc

    forecast_payload = None
    forecast_average_payload = None
    if include_forecast:
        try:
            horizons = _parse_forecast_horizons(forecast_horizons)
            resolved_lookback = _resolve_forecast_lookback_hours(horizons, forecast_lookback_hours)
            sensor_warning_map = {
                feature: details.get("sensor_warning")
                for feature, details in (feature_assessment or {}).items()
                if isinstance(details, dict) and details.get("sensor_warning")
            }
            forecast_payload = build_forecast(
                base_feature_values=feature_values,
                horizons_hours=horizons,
                lookback_hours=resolved_lookback,
                sensor_warning_map=sensor_warning_map,
            )
            forecast_average_payload = summarize_forecast_average(forecast_payload)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    model_names = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "svm": "SVM",
    }
    prediction_message = f"Predicție realizată cu modelul {model_names[model_type]}."
    if use_hourly_average:
        prediction_message = (
            f"Predicție realizată cu modelul {model_names[model_type]} folosind medii agregate pe interval orar."
        )

    algorithm_comparison = None
    if compare_models:
        algorithm_comparison = []
        for candidate_model in model_names:
            try:
                candidate_prediction, candidate_confidence, _, _ = predict_air_quality(
                    use_hourly_average=use_hourly_average,
                    aggregation_hours=aggregation_hours,
                    model_type=candidate_model,
                )
                algorithm_comparison.append({
                    "model_type": candidate_model,
                    "model_name": model_names[candidate_model],
                    "prediction": str(candidate_prediction),
                    "confidence": candidate_confidence,
                    "status": "success",
                })
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                algorithm_comparison.append({
                    "model_type": candidate_model,
                    "model_name": model_names[candidate_model],
                    "status": "unavailable",
                    "error": str(exc),
                })

    response_payload = PredictionResponse(
        status="success",
        message=prediction_message,
        prediction=str(prediction),
        confidence=confidence,
        model_type=model_type,
        algorithm_comparison=algorithm_comparison,
        input_values=feature_values,
        feature_assessment=feature_assessment,
        source_measurement=feature_values,
        forecast=forecast_payload,
        forecast_average=forecast_average_payload,
    )

    app.state.latest_prediction = response_payload.model_dump()
    return response_payload


@app.post("/predict-demo", response_model=PredictionResponse)
def predict_demo():
    """Predicție cu date de test (demo) - util pentru testing."""
    try:
        # Date de test pentru demo
        demo_data = {
            "temperature": 22.5,
            "humidity": 55.0,
            "pm25": 18.5,
            "pm10": 35.2,
            "co2": 950.0
        }
        
        from app.models.train_model import load_model
        model = load_model()
        
        # Crează DataFrame cu datele demo
        input_df = pd.DataFrame([demo_data])
        prediction = model.predict(input_df)[0]
        confidence = float(np.max(model.predict_proba(input_df)))
        
        from app.services.predictor import _build_feature_assessment
        feature_assessment = _build_feature_assessment(demo_data, sensor_warning_map={})
        
        response_payload = PredictionResponse(
            status="success",
            message="Predicție de test cu date sample - Date: T=22.5°C, H=55%, PM2.5=18.5, PM10=35.2, CO2=950ppm",
            prediction=str(prediction),
            confidence=confidence,
            input_values=demo_data,
            feature_assessment=feature_assessment,
            source_measurement=demo_data,
            forecast=None,
        )
        
        return response_payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Eroare la predicția demo: {str(exc)}") from exc


class CustomPredictionRequest(BaseModel):
    temperature: float = Field(ge=-50, le=60)
    humidity: float = Field(ge=0, le=100)
    pm25: float
    pm10: float
    co2: float = Field(ge=400, le=5000)


@app.post("/predict-custom", response_model=PredictionResponse)
def predict_custom(data: CustomPredictionRequest):
    """Predicție cu date custom furnizate în request."""
    try:
        from app.models.train_model import load_model
        
        feature_values = {
            "temperature": data.temperature,
            "humidity": data.humidity,
            "pm25": data.pm25,
            "pm10": data.pm10,
            "co2": data.co2
        }
        
        model = load_model()
        input_df = pd.DataFrame([feature_values])
        prediction = model.predict(input_df)[0]
        confidence = float(np.max(model.predict_proba(input_df)))
        
        from app.services.predictor import _build_feature_assessment
        feature_assessment = _build_feature_assessment(feature_values, sensor_warning_map={})
        
        response_payload = PredictionResponse(
            status="success",
            message="Predicție realizată cu date custom furnizate.",
            prediction=str(prediction),
            confidence=confidence,
            input_values=feature_values,
            feature_assessment=feature_assessment,
            source_measurement=feature_values,
            forecast=None,
        )
        
        return response_payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Eroare la predicția custom: {str(exc)}") from exc



@app.post("/train")
def train_model(request: TrainRequest):
    try:
        if request.training_model == "isolation_forest":
            _, training_report = train_and_save_isolation_forest(
                return_report=True,
                use_hourly_aggregation=True,
                aggregation_hours=request.aggregation_hours,
                aggregation_minutes=request.aggregation_minutes,
            )
            trained_model = "isolation_forest"
        elif request.training_model == "svm":
            _, training_report = train_and_save_svm(
                return_report=True,
                use_hourly_aggregation=True,
                aggregation_hours=request.aggregation_hours,
                aggregation_minutes=request.aggregation_minutes,
                allow_derived_label_fallback=request.allow_derived_label_fallback,
            )
            trained_model = "svm"
        elif request.training_model == "xgboost":
            _, training_report = train_and_save_xgboost(
                return_report=True,
                use_hourly_aggregation=True,
                aggregation_hours=request.aggregation_hours,
                aggregation_minutes=request.aggregation_minutes,
                allow_derived_label_fallback=request.allow_derived_label_fallback,
            )
            trained_model = "xgboost"
        else:
            _, training_report = train_and_save_random_forest(
                return_report=True,
                use_hourly_aggregation=True,
                aggregation_hours=request.aggregation_hours,
                aggregation_minutes=request.aggregation_minutes,
                allow_derived_label_fallback=request.allow_derived_label_fallback,
            )
            trained_model = "random_forest"
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Eroare la antrenare: {str(exc)}") from exc

    if request.aggregation_minutes is not None:
        training_message = (
            "Modelul a fost antrenat și salvat folosind date agregate pe intervalul selectat în minute."
        )
    else:
        training_message = (
            "Modelul a fost antrenat și salvat folosind agregare la nivel de minut pe intervalul orar selectat."
        )

    response_payload = {
        "status": "success",
        "message": training_message,
        "model_type": trained_model,
        "training_report": training_report,
        "dataset_name": request.dataset_name,
        "notes": request.notes,
        "allow_derived_label_fallback": request.allow_derived_label_fallback,
    }

    if request.allow_derived_label_fallback:
        response_payload["warning"] = (
            "Antrenarea permite fallback la etichete derivate din feature-uri. "
            "Acest mod poate introduce data leakage si nu este potrivit pentru evaluare de performanta."
        )

    app.state.latest_training = response_payload
    return response_payload


@app.post("/train-demo")
def train_demo(allow_derived_label_fallback: bool = Query(default=True)):
    """Antrenare demo cu date de test - util pentru testing.

    `allow_derived_label_fallback` implicit True (comportament demo istoric), dar poate fi
    dezactivat explicit din UI pentru a nu introduce data leakage.
    """
    try:
        # Antrenează modelul cu datele disponibile din bază
        _, training_report = train_and_save_random_forest(
            return_report=True,
            use_hourly_aggregation=True,
            aggregation_hours=24,
            aggregation_minutes=None,
            allow_derived_label_fallback=allow_derived_label_fallback,
        )
        
        response_payload = {
            "status": "success",
            "message": "Model Random Forest antrenat cu succes pe date demo (24h agregare)",
            "model_type": "random_forest",
            "training_report": training_report,
            "dataset_name": "demo-24h",
            "notes": "Antrenare demo - folosit pentru testing",
            "allow_derived_label_fallback": allow_derived_label_fallback,
        }

        if allow_derived_label_fallback:
            response_payload["warning"] = (
                "Antrenarea permite fallback la etichete derivate din feature-uri. "
                "Acest mod poate introduce data leakage si nu este potrivit pentru evaluare de performanta."
            )

        app.state.latest_training = response_payload
        return response_payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Eroare la antrenare demo: {str(exc)}") from exc


@app.post("/anomaly")
def detect_anomaly():
    try:
        result = detect_anomaly_service()

        return {
            "status": "success",
            "message": "Detecție realizată cu modelul Isolation Forest folosind ultima înregistrare din measurements.",
            "sensor_id": "latest-measurement",
            "result": result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Eroare la detecția de anomalii: {str(exc)}") from exc


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    model_outputs = {
        "latest_prediction": getattr(app.state, "latest_prediction", None),
        "latest_training": getattr(app.state, "latest_training", None),
    }
    reply = get_chatbot_reply(
        request.message,
        model_outputs=model_outputs,
        conversation_history=[chat_message.model_dump() for chat_message in request.history],
    )
    return ChatResponse(reply=reply)
