from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal, Optional
import pandas as pd
import numpy as np
import requests
from pathlib import Path

from app.models.train_model import (
    train_and_save_isolation_forest,
    train_and_save_random_forest,
    train_and_save_svm,
)
from app.models.xgboost_model import train_and_save_xgboost
from app.core.database import get_device_identifiers, get_devices_with_location, get_measurements
from app.services.chatbot import get_chatbot_reply_details, get_chatbot_welcome_message
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
app.state.cargo_profiles = {}

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
    device_identifier: Optional[str] = None


class CargoTemperaturePolicy(BaseModel):
    product_name: str = Field(min_length=1, max_length=120, description="Produsul transportat.")
    min_temperature: float = Field(ge=-80, le=80, description="Limita minima furnizata de operator.")
    max_temperature: float = Field(ge=-80, le=80, description="Limita maxima furnizata de operator.")
    min_humidity: Optional[float] = Field(default=None, ge=0, le=100)
    max_humidity: Optional[float] = Field(default=None, ge=0, le=100)
    target_temperature: Optional[float] = Field(
        default=None,
        ge=-80,
        le=80,
        description="Setpoint recomandat de operator; implicit este mijlocul intervalului.",
    )

    @model_validator(mode="after")
    def validate_ranges(self):
        if self.min_temperature > self.max_temperature:
            raise ValueError("min_temperature trebuie sa fie <= max_temperature.")
        if self.min_humidity is not None and self.max_humidity is not None and self.min_humidity > self.max_humidity:
            raise ValueError("min_humidity trebuie sa fie <= max_humidity.")
        if self.target_temperature is not None and not self.min_temperature <= self.target_temperature <= self.max_temperature:
            raise ValueError("target_temperature trebuie sa fie in intervalul de temperatura.")
        return self


class ParameterLimit(BaseModel):
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    @model_validator(mode="after")
    def validate_limits(self):
        if self.min_value is None and self.max_value is None:
            raise ValueError("Trebuie configurata cel putin o limita: min_value sau max_value.")
        if self.min_value is not None and self.max_value is not None and self.min_value > self.max_value:
            raise ValueError("min_value trebuie sa fie <= max_value.")
        return self


class CargoAssessmentRequest(BaseModel):
    profile_id: Optional[str] = Field(default=None, description="Profil de transport salvat anterior.")
    product_name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    min_temperature: Optional[float] = Field(default=None, ge=-80, le=80)
    max_temperature: Optional[float] = Field(default=None, ge=-80, le=80)
    min_humidity: Optional[float] = Field(default=None, ge=0, le=100)
    max_humidity: Optional[float] = Field(default=None, ge=0, le=100)
    target_temperature: Optional[float] = Field(default=None, ge=-80, le=80)
    device_identifier: Optional[str] = Field(default=None, description="Dispozitivul din care se citesc valorile.")
    temperature: Optional[float] = Field(default=None, ge=-80, le=80)
    humidity: Optional[float] = Field(default=None, ge=0, le=100)
    parameter_limits: dict[str, ParameterLimit] = Field(
        default_factory=dict,
        description="Limite min/max configurate pentru orice parametru senzorial.",
    )
    parameter_values: dict[str, float] = Field(
        default_factory=dict,
        description="Valori senzoriale trimise direct de client pentru parametrii configurati.",
    )
    research_product: Optional[bool] = Field(default=None)

    @model_validator(mode="after")
    def validate_inline_policy(self):
        if self.profile_id:
            return self
        if not self.product_name or self.min_temperature is None or self.max_temperature is None:
            raise ValueError(
                "Trimite profile_id sau configureaza product_name, min_temperature si max_temperature."
            )
        if self.min_temperature > self.max_temperature:
            raise ValueError("min_temperature trebuie sa fie <= max_temperature.")
        if self.min_humidity is not None and self.max_humidity is not None and self.min_humidity > self.max_humidity:
            raise ValueError("min_humidity trebuie sa fie <= max_humidity.")
        return self


class CargoAssessmentResponse(BaseModel):
    status: Literal["within_limits", "too_cold", "too_hot", "humidity_out_of_range", "missing_measurement"]
    message: str
    product_name: str
    device_identifier: Optional[str] = None
    measured_temperature: Optional[float] = None
    measured_humidity: Optional[float] = None
    policy: dict[str, Any]
    recommended_temperature: float
    recommended_action: str
    alerts: list[str] = Field(default_factory=list)
    parameter_status: dict[str, str] = Field(default_factory=dict)
    parameter_values: dict[str, float] = Field(default_factory=dict)
    profile_id: Optional[str] = None
    product_semantic_term: Optional[str] = None
    product_research: dict[str, Any] | None = None


class CargoProfile(BaseModel):
    profile_id: str = Field(min_length=1, max_length=100)
    customer_id: Optional[str] = Field(default=None, max_length=100)
    product_name: str = Field(min_length=1, max_length=120)
    min_temperature: float = Field(ge=-80, le=80)
    max_temperature: float = Field(ge=-80, le=80)
    min_humidity: Optional[float] = Field(default=None, ge=0, le=100)
    max_humidity: Optional[float] = Field(default=None, ge=0, le=100)
    target_temperature: Optional[float] = Field(default=None, ge=-80, le=80)
    parameter_limits: dict[str, ParameterLimit] = Field(default_factory=dict)
    research_product: bool = False

    @model_validator(mode="after")
    def validate_profile(self):
        if self.min_temperature > self.max_temperature:
            raise ValueError("min_temperature trebuie sa fie <= max_temperature.")
        if self.min_humidity is not None and self.max_humidity is not None and self.min_humidity > self.max_humidity:
            raise ValueError("min_humidity trebuie sa fie <= max_humidity.")
        return self


PRODUCT_SEMANTIC_ALIASES = {
    "banane": "banana", "banana": "banana", "mere": "apple", "măr": "apple",
    "merele": "apple", "portocale": "orange", "portocală": "orange",
    "struguri": "grape", "căpșuni": "strawberry", "capsuni": "strawberry",
    "avocado": "avocado", "kiwi": "kiwi", "piersici": "peach", "pere": "pear",
    "pepene": "watermelon", "cireșe": "cherry", "cirese": "cherry",
}


def _semantic_product_term(product_name: str) -> str:
    normalized = product_name.strip().casefold()
    return next(
        (semantic for alias, semantic in PRODUCT_SEMANTIC_ALIASES.items() if alias in normalized),
        normalized,
    )


def _research_product_transport(product_name: str, semantic_term: str) -> dict[str, Any]:
    query = f"{product_name} ({semantic_term}) transport storage temperature humidity fruit"
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return {"status": "unavailable", "query": query, "results": []}

    results: list[dict[str, str]] = []
    abstract = (payload.get("AbstractText") or "").strip()
    if abstract:
        results.append({"text": abstract, "url": (payload.get("AbstractURL") or "").strip()})
    for topic in payload.get("RelatedTopics") or []:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({"text": topic["Text"].strip(), "url": (topic.get("FirstURL") or "").strip()})
        if len(results) >= 3:
            break
    return {"status": "found" if results else "no_results", "query": query, "results": results}


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


def _extract_numeric_value(row: pd.Series, candidates: tuple[str, ...]) -> float | None:
    for candidate in candidates:
        if candidate not in row.index:
            continue
        value = pd.to_numeric(row.get(candidate), errors="coerce")
        if pd.notna(value):
            return float(value)
    return None


class TrainRequest(BaseModel):
    dataset_name: Optional[str] = None
    notes: Optional[str] = None
    training_model: str = "random_forest"
    device_identifier: Optional[str] = Field(
        default=None,
        description="Antreneaza doar pe datele dispozitivului selectat. Gol = toate dispozitivele.",
    )
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
    message: Optional[str] = Field(default=None, min_length=1, max_length=4_000)
    text: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=4_000,
        description="Alias pentru 'message' folosit de unele integrari externe.",
    )
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)
    device_identifier: Optional[str] = Field(
        default=None,
        description="Raspunde folosind doar datele dispozitivului selectat.",
    )
    selected: Optional[str] = Field(
        default=None,
        description="Alias pentru 'device_identifier' folosit de unele integrari externe.",
    )
    session_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Identificator sesiune client pentru gestionarea contextului la schimbarea dispozitivului.",
    )


class ChatResponse(BaseModel):
    reply: str
    text: str
    selected: Optional[str] = None
    device_changed: bool = False
    device_change_message: Optional[str] = None
    models_used: list[str] = Field(default_factory=list)
    bert_available: bool = False
    bert_model: Optional[str] = None
    bert_feature: Optional[str] = None
    response_engine: str = "rule_based_chatbot"


class ChatWelcomeResponse(BaseModel):
    message: str
    text: str


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


@app.post("/transport/profiles", response_model=CargoProfile)
def create_cargo_profile(profile: CargoProfile):
    """Create or replace a reusable cargo transport profile."""
    supported_parameters = {"temperature", "humidity", "pm25", "pm10", "co2", "voc"}
    unknown_parameters = set(profile.parameter_limits) - supported_parameters
    if unknown_parameters:
        raise HTTPException(
            status_code=422,
            detail=f"Parametri necunoscuti: {', '.join(sorted(unknown_parameters))}.",
        )
    app.state.cargo_profiles[profile.profile_id] = profile.model_dump()
    return profile


@app.get("/transport/profiles", response_model=list[CargoProfile])
def list_cargo_profiles(customer_id: str | None = Query(default=None)):
    profiles = list(app.state.cargo_profiles.values())
    if customer_id is not None:
        profiles = [profile for profile in profiles if profile.get("customer_id") == customer_id]
    return profiles


@app.get("/transport/profiles/{profile_id}", response_model=CargoProfile)
def get_cargo_profile(profile_id: str):
    profile = app.state.cargo_profiles.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profilul de transport nu există.")
    return profile


@app.post("/transport/cargo-assessment", response_model=CargoAssessmentResponse)
def assess_cargo_transport(request: CargoAssessmentRequest):
    """Evaluate cargo conditions using operator-supplied product limits.

    The API intentionally does not contain hardcoded fruit requirements. The
    carrier or their technical specialist supplies the approved transport range.
    """
    profile = None
    if request.profile_id:
        profile = app.state.cargo_profiles.get(request.profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Profilul de transport nu există.")

    product_name = profile["product_name"] if profile else request.product_name
    semantic_term = _semantic_product_term(product_name)
    research_enabled = profile.get("research_product", False) if profile else bool(request.research_product)
    product_research = _research_product_transport(product_name, semantic_term) if research_enabled else None
    min_temperature = profile["min_temperature"] if profile else request.min_temperature
    max_temperature = profile["max_temperature"] if profile else request.max_temperature
    min_humidity = profile.get("min_humidity") if profile else request.min_humidity
    max_humidity = profile.get("max_humidity") if profile else request.max_humidity
    target_temperature = profile.get("target_temperature") if profile else request.target_temperature
    profile_limits = profile.get("parameter_limits", {}) if profile else request.parameter_limits

    measured_temperature = request.temperature
    measured_humidity = request.humidity
    resolved_device_identifier = (request.device_identifier or "").strip() or None

    supported_parameters = {"temperature", "humidity", "pm25", "pm10", "co2", "voc"}
    unknown_parameters = set(profile_limits) - supported_parameters
    if unknown_parameters:
        raise HTTPException(
            status_code=422,
            detail=f"Parametri necunoscuti: {', '.join(sorted(unknown_parameters))}.",
        )

    configured_limits = {
        name: ParameterLimit(**values) if isinstance(values, dict) else values
        for name, values in profile_limits.items()
    }
    if "temperature" not in configured_limits:
        configured_limits["temperature"] = ParameterLimit(
            min_value=min_temperature,
            max_value=max_temperature,
        )
    if min_humidity is not None or max_humidity is not None:
        configured_limits.setdefault(
            "humidity",
            ParameterLimit(min_value=min_humidity, max_value=max_humidity),
        )

    parameter_values = dict(request.parameter_values)
    if measured_temperature is None:
        measured_temperature = parameter_values.get("temperature")
    if measured_humidity is None:
        measured_humidity = parameter_values.get("humidity")
    if measured_temperature is not None:
        parameter_values["temperature"] = measured_temperature
    if measured_humidity is not None:
        parameter_values["humidity"] = measured_humidity

    if measured_temperature is None or measured_humidity is None:
        try:
            measurements = get_measurements(
                device_identifier=resolved_device_identifier,
                limit=1,
                descending=True,
                raise_on_error=True,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not measurements.empty:
            latest = measurements.iloc[0]
            if measured_temperature is None:
                measured_temperature = _extract_numeric_value(latest, ("temperature", "temperatura", "temp"))
            if measured_humidity is None:
                measured_humidity = _extract_numeric_value(latest, ("humidity", "umiditate"))
            for parameter_name, candidates in {
                "pm25": ("pm25", "pm2_5", "pm2.5"),
                "pm10": ("pm10",),
                "co2": ("co2", "co_2"),
                "voc": ("voc", "tvoc"),
            }.items():
                if parameter_name not in parameter_values:
                    value = _extract_numeric_value(latest, candidates)
                    if value is not None:
                        parameter_values[parameter_name] = value

    if measured_temperature is not None:
        parameter_values["temperature"] = measured_temperature
    if measured_humidity is not None:
        parameter_values["humidity"] = measured_humidity

    temperature_limits = configured_limits["temperature"]
    effective_min_temperature = temperature_limits.min_value
    effective_max_temperature = temperature_limits.max_value
    recommended_temperature = target_temperature
    if recommended_temperature is None:
        recommended_temperature = (
            (effective_min_temperature or min_temperature)
            + (effective_max_temperature or max_temperature)
        ) / 2

    policy_payload = {
        "min_temperature": min_temperature,
        "max_temperature": max_temperature,
        "min_humidity": min_humidity,
        "max_humidity": max_humidity,
        "target_temperature": recommended_temperature,
        "parameter_limits": {
            parameter_name: limit.model_dump()
            for parameter_name, limit in configured_limits.items()
        },
    }

    if measured_temperature is None:
        return CargoAssessmentResponse(
            status="missing_measurement",
            message="Nu există o temperatură măsurată pentru evaluarea transportului.",
            product_name=product_name,
            device_identifier=resolved_device_identifier,
            policy=policy_payload,
            recommended_temperature=recommended_temperature,
            recommended_action="Trimite temperatura senzorului sau configurează un device_identifier valid.",
            profile_id=request.profile_id,
            product_semantic_term=semantic_term,
            product_research=product_research,
        )

    alerts: list[str] = []
    parameter_status: dict[str, str] = {}
    if effective_min_temperature is not None and measured_temperature < effective_min_temperature:
        status = "too_cold"
        alerts.append("Temperatura este sub limita configurată pentru produs.")
        action = "Crește temperatura treptat spre setpointul recomandat și verifică riscul de îngheț."
    elif effective_max_temperature is not None and measured_temperature > effective_max_temperature:
        status = "too_hot"
        alerts.append("Temperatura este peste limita configurată pentru produs.")
        action = "Activează răcirea și readu temperatura treptat spre setpointul recomandat."
    else:
        status = "within_limits"
        action = "Menține temperatura în intervalul configurat și continuă monitorizarea."

    parameter_status["temperature"] = status

    if measured_humidity is not None:
        humidity_out = (
            min_humidity is not None and measured_humidity < min_humidity
        ) or (
            max_humidity is not None and measured_humidity > max_humidity
        )
        if humidity_out:
            alerts.append("Umiditatea este în afara limitelor configurate.")
            if status == "within_limits":
                status = "humidity_out_of_range"
            parameter_status["humidity"] = "out_of_range"
        else:
            parameter_status["humidity"] = "within_limits"

    for parameter_name, limits in configured_limits.items():
        value = parameter_values.get(parameter_name)
        if value is None or parameter_name in {"temperature", "humidity"}:
            continue
        if limits.min_value is not None and value < limits.min_value:
            parameter_status[parameter_name] = "below_min"
            alerts.append(f"{parameter_name} este sub limita minima configurata.")
        elif limits.max_value is not None and value > limits.max_value:
            parameter_status[parameter_name] = "above_max"
            alerts.append(f"{parameter_name} este peste limita maxima configurata.")
        else:
            parameter_status[parameter_name] = "within_limits"

    return CargoAssessmentResponse(
        status=status,
        message=f"Evaluare transport pentru {product_name}: {status}.",
        product_name=product_name,
        device_identifier=resolved_device_identifier,
        measured_temperature=measured_temperature,
        measured_humidity=measured_humidity,
        policy=policy_payload,
        recommended_temperature=recommended_temperature,
        recommended_action=action,
        alerts=alerts,
        parameter_status=parameter_status,
        parameter_values=parameter_values,
        profile_id=request.profile_id,
        product_semantic_term=semantic_term,
        product_research=product_research,
    )


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/devices")
def list_devices():
    try:
        device_details = get_devices_with_location()
        devices = [item["device_identifier"] for item in device_details]
        if not devices:
            devices = get_device_identifiers()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"status": "success", "devices": devices, "device_details": device_details}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    model_type: Literal["random_forest", "xgboost", "svm"] = Query(default="random_forest"),
    compare_models: bool = Query(default=False),
    use_hourly_average: bool = Query(default=False),
    aggregation_hours: int = Query(default=1, ge=1, le=168),
    include_forecast: bool = Query(default=False),
    forecast_horizons: str = Query(default="1,3,6,12,24"),
    forecast_lookback_hours: int | None = Query(default=None, ge=2, le=720),
    device_identifier: str | None = Query(
        default=None,
        description="Foloseste doar datele dispozitivului selectat. Gol = toate dispozitivele.",
    ),
):
    try:
        prediction, confidence, feature_values, feature_assessment = predict_air_quality(
            use_hourly_average=use_hourly_average,
            aggregation_hours=aggregation_hours,
            model_type=model_type,
            device_identifier=device_identifier,
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
                device_identifier=device_identifier,
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
    if device_identifier:
        prediction_message += f" Date filtrate pentru dispozitivul '{device_identifier}'."

    algorithm_comparison = None
    if compare_models:
        algorithm_comparison = []
        for candidate_model in model_names:
            try:
                candidate_prediction, candidate_confidence, _, _ = predict_air_quality(
                    use_hourly_average=use_hourly_average,
                    aggregation_hours=aggregation_hours,
                    model_type=candidate_model,
                    device_identifier=device_identifier,
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
        device_identifier=device_identifier,
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
    device_identifier = (request.device_identifier or "").strip() or None
    try:
        if request.training_model == "isolation_forest":
            _, training_report = train_and_save_isolation_forest(
                return_report=True,
                use_hourly_aggregation=True,
                aggregation_hours=request.aggregation_hours,
                aggregation_minutes=request.aggregation_minutes,
                device_identifier=device_identifier,
            )
            trained_model = "isolation_forest"
        elif request.training_model == "svm":
            _, training_report = train_and_save_svm(
                return_report=True,
                use_hourly_aggregation=True,
                aggregation_hours=request.aggregation_hours,
                aggregation_minutes=request.aggregation_minutes,
                allow_derived_label_fallback=request.allow_derived_label_fallback,
                device_identifier=device_identifier,
            )
            trained_model = "svm"
        elif request.training_model == "xgboost":
            _, training_report = train_and_save_xgboost(
                return_report=True,
                use_hourly_aggregation=True,
                aggregation_hours=request.aggregation_hours,
                aggregation_minutes=request.aggregation_minutes,
                allow_derived_label_fallback=request.allow_derived_label_fallback,
                device_identifier=device_identifier,
            )
            trained_model = "xgboost"
        else:
            _, training_report = train_and_save_random_forest(
                return_report=True,
                use_hourly_aggregation=True,
                aggregation_hours=request.aggregation_hours,
                aggregation_minutes=request.aggregation_minutes,
                allow_derived_label_fallback=request.allow_derived_label_fallback,
                device_identifier=device_identifier,
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

    if device_identifier:
        training_message += f" Au fost folosite doar datele dispozitivului '{device_identifier}'."

    response_payload = {
        "status": "success",
        "message": training_message,
        "model_type": trained_model,
        "training_report": training_report,
        "dataset_name": request.dataset_name,
        "notes": request.notes,
        "device_identifier": device_identifier,
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
def detect_anomaly(
    device_identifier: str | None = Query(
        default=None,
        description="Foloseste doar datele dispozitivului selectat. Gol = toate dispozitivele.",
    ),
):
    try:
        result = detect_anomaly_service(device_identifier=device_identifier)

        message = (
            "Detecție realizată cu modelul Isolation Forest folosind ultima înregistrare din measurements."
        )
        if device_identifier:
            message += f" Dispozitiv selectat: '{device_identifier}'."

        return {
            "status": "success",
            "message": message,
            "sensor_id": device_identifier or "latest-measurement",
            "device_identifier": device_identifier,
            "result": result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Eroare la detecția de anomalii: {str(exc)}") from exc


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    message = (request.message or request.text or "").strip()
    resolved_device_identifier = (request.device_identifier or request.selected or "").strip() or None
    session_id = (request.session_id or "default").strip()

    if not hasattr(app.state, "chat_device_by_session"):
        app.state.chat_device_by_session = {}

    previous_device_identifier = app.state.chat_device_by_session.get(session_id)
    device_changed = previous_device_identifier is not None and previous_device_identifier != resolved_device_identifier
    app.state.chat_device_by_session[session_id] = resolved_device_identifier

    device_change_message = None
    if device_changed:
        previous_label = previous_device_identifier or "toate dispozitivele"
        current_label = resolved_device_identifier or "toate dispozitivele"
        device_change_message = (
            f"Context resetat: dispozitiv schimbat din '{previous_label}' în '{current_label}'."
        )

    if not message:
        welcome_message = get_chatbot_welcome_message()
        if device_change_message:
            welcome_message = f"{welcome_message}\n\n{device_change_message}"
        return ChatResponse(
            reply=welcome_message,
            text=welcome_message,
            selected=resolved_device_identifier,
            device_changed=device_changed,
            device_change_message=device_change_message,
            models_used=["rule_based_chatbot"],
            response_engine="rule_based_chatbot",
        )

    model_outputs = {
        "latest_prediction": getattr(app.state, "latest_prediction", None),
        "latest_training": getattr(app.state, "latest_training", None),
    }
    reply_details = {
        "models_used": ["rule_based_chatbot"],
        "bert_available": False,
        "bert_model": None,
        "bert_feature": None,
        "response_engine": "rule_based_chatbot",
    }
    try:
        reply_details = get_chatbot_reply_details(
            message,
            model_outputs=model_outputs,
            conversation_history=[] if device_changed else [chat_message.model_dump() for chat_message in request.history],
            device_identifier=resolved_device_identifier,
        )
        reply = reply_details["reply"]
    except Exception:
        # Never bubble up a 500 here: a raw HTTP error makes the mobile client
        # show a generic "AI indisponibil" message far more often than needed.
        reply = (
            "Nu am putut procesa complet cererea acum, dar sunt disponibil. "
            "Poți reformula întrebarea sau încearcă din nou în câteva secunde."
        )
    if device_change_message:
        reply = f"{device_change_message}\n\n{reply}"
    return ChatResponse(
        reply=reply,
        text=reply,
        selected=resolved_device_identifier,
        device_changed=device_changed,
        device_change_message=device_change_message,
        models_used=reply_details.get("models_used", []),
        bert_available=reply_details.get("bert_available", False),
        bert_model=reply_details.get("bert_model"),
        bert_feature=reply_details.get("bert_feature"),
        response_engine=reply_details.get("response_engine", "rule_based_chatbot"),
    )


@app.get("/chat/welcome", response_model=ChatWelcomeResponse)
def chat_welcome():
    welcome_message = get_chatbot_welcome_message()
    return ChatWelcomeResponse(message=welcome_message, text=welcome_message)
