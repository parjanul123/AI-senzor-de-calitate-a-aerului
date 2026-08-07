from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any, Literal, Optional

from app.models.train_model import (
    train_and_save_isolation_forest,
    train_and_save_model,
    train_and_save_random_forest,
    train_and_save_svm,
)
from app.models.xgboost_model import train_and_save_xgboost
from app.services.chatbot import get_chatbot_reply
from app.services.anomaly_detector import detect_anomaly as detect_anomaly_service
from app.services.predictor import build_forecast, predict_air_quality

app = FastAPI(
    title="Air Quality AI API",
    version="1.0.0",
    description="Backend pentru monitorizarea calității aerului și integrarea cu modele AI"
)

app.state.latest_prediction = None
app.state.latest_training = None


class PredictionResponse(BaseModel):
    status: str
    message: str
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    input_values: dict[str, float] | None = None
    feature_assessment: dict[str, Any] | None = None
    source_measurement: dict[str, Any] | None = None
    forecast: list[dict[str, Any]] | None = None


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


@app.get("/")
def root():
    return {
        "service": "air-quality-ai-api",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
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
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    forecast_payload = None
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
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    prediction_message = "Predicție realizată cu modelul Random Forest."
    if use_hourly_average:
        prediction_message = (
            "Predicție realizată cu modelul Random Forest folosind medii agregate pe interval orar."
        )

    response_payload = PredictionResponse(
        status="success",
        message=prediction_message,
        prediction=str(prediction),
        confidence=confidence,
        input_values=feature_values,
        feature_assessment=feature_assessment,
        source_measurement=feature_values,
        forecast=forecast_payload,
    )

    app.state.latest_prediction = response_payload.model_dump()
    return response_payload


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
                allow_derived_label_fallback=True,
            )
            trained_model = "svm"
        elif request.training_model == "xgboost":
            _, training_report = train_and_save_xgboost(
                return_report=True,
                use_hourly_aggregation=True,
                aggregation_hours=request.aggregation_hours,
                aggregation_minutes=request.aggregation_minutes,
                allow_derived_label_fallback=True,
            )
            trained_model = "xgboost"
        else:
            _, training_report = train_and_save_random_forest(
                return_report=True,
                use_hourly_aggregation=True,
                aggregation_hours=request.aggregation_hours,
                aggregation_minutes=request.aggregation_minutes,
                allow_derived_label_fallback=True,
            )
            trained_model = "random_forest"
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
    }

    app.state.latest_training = response_payload
    return response_payload


@app.post("/anomaly")
def detect_anomaly():
    result = detect_anomaly_service()

    return {
        "status": "success",
        "message": "Detecție realizată cu modelul Isolation Forest folosind ultima înregistrare din measurements.",
        "sensor_id": "latest-measurement",
        "result": result,
    }


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
