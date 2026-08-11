import numpy as np
import pandas as pd

from app.core.database import get_measurements
from app.models.train_model import load_model

FEATURE_ALIASES = {
    "temperature": ["temperature", "temperatura", "temp"],
    "humidity": ["humidity", "umiditate"],
    "pm25": ["pm25", "pm2_5", "pm2.5"],
    "pm10": ["pm10"],
    "co2": ["co2", "co_2"],
}
TIMESTAMP_CANDIDATES = ["created_at", "timestamp", "time", "recorded_at"]

FEATURE_DISPLAY_NAMES = {
    "temperature": "Temperatura",
    "humidity": "Umiditate",
    "pm25": "PM2.5",
    "pm10": "PM10",
    "co2": "CO2",
}

FEATURE_UNITS = {
    "temperature": "°C",
    "humidity": "%",
    "pm25": "µg/m3",
    "pm10": "µg/m3",
    "co2": "ppm",
}

HUMIDITY_RANGE = (0.0, 100.0)
MONTHLY_TEMPERATURE_RANGES = {
    1: (-20.0, 22.0), 2: (-18.0, 25.0), 3: (-10.0, 30.0),
    4: (-2.0, 35.0), 5: (3.0, 38.0), 6: (8.0, 42.0),
    7: (10.0, 45.0), 8: (10.0, 45.0), 9: (3.0, 40.0),
    10: (-3.0, 34.0), 11: (-10.0, 28.0), 12: (-18.0, 24.0),
}
LOCAL_TIMEZONE = "Europe/Bucharest"
MAX_TEMPERATURE_CHANGE_PER_HOUR = 0.25
MIN_CALENDAR_PROFILE_SAMPLES = 3
ZERO_WARNING_MIN_SAMPLES = 24
ZERO_WARNING_RATIO_THRESHOLD = 0.6
ZERO_WARNING_STREAK_THRESHOLD = 8
ZERO_WARNING_SUDDEN_DROP_STREAK_THRESHOLD = 3
ZERO_WARNING_LOOKBACK_ROWS = 240


def _human_condition_label(feature_name: str, value: float) -> str:
    if feature_name == "temperature":
        if value < 18.0:
            return "prea rece"
        if value <= 26.0:
            return "bine"
        if value <= 30.0:
            return "cald"
        return "foarte cald"

    if feature_name == "humidity":
        if value < 30.0:
            return "prea uscat"
        if value <= 60.0:
            return "confortabil"
        if value <= 70.0:
            return "usor umed"
        return "prea umed"

    if feature_name == "co2":
        if value <= 800.0:
            return "nepoluat"
        if value <= 1200.0:
            return "aer incarcat"
        return "poluat"

    if feature_name == "pm25":
        if value <= 15.0:
            return "scazut"
        if value <= 35.0:
            return "moderat"
        return "ridicat"

    if feature_name == "pm10":
        if value <= 45.0:
            return "scazut"
        if value <= 90.0:
            return "moderat"
        return "ridicat"

    return "necunoscut"

# Praguri orientative pentru expunere umana in spatii interioare.
FEATURE_HUMAN_BANDS = {
    "temperature": {
        "good": (20.0, 26.0, "Confort termic pentru majoritatea persoanelor."),
        "moderate": (17.0, 30.0, "Poate provoca disconfort termic (prea rece/prea cald)."),
        "poor": (None, None, "Risc crescut de stres termic si scaderea confortului."),
    },
    "humidity": {
        "good": (40.0, 60.0, "Interval recomandat pentru confort respirator."),
        "moderate": (30.0, 70.0, "Poate cauza uscaciune sau senzatie de aer incarcat."),
        "poor": (None, None, "Risc mai mare de iritatii respiratorii si mucegai/aer uscat sever."),
    },
    "pm25": {
        "good": (None, 15.0, "Impact redus pentru majoritatea persoanelor sanatoase."),
        "moderate": (15.0, 35.0, "Persoanele sensibile pot simti iritatie respiratorie usoara."),
        "poor": (35.0, None, "Risc ridicat de iritatii respiratorii si impact cardiovascular la expunere."),
    },
    "pm10": {
        "good": (None, 45.0, "Nivel in general acceptabil pentru expunere zilnica."),
        "moderate": (45.0, 90.0, "Poate produce disconfort la persoanele sensibile."),
        "poor": (90.0, None, "Risc crescut de simptome respiratorii la expunere."),
    },
    "co2": {
        "good": (None, 800.0, "Calitate buna a ventilatiei, disconfort minim."),
        "moderate": (800.0, 1200.0, "Poate provoca somnolenta usoara sau scaderea atentiei."),
        "poor": (1200.0, None, "Ventilatie insuficienta, risc de cefalee, oboseala si scaderea concentrarii."),
    },
}


def _extract_measurement_feature(row: pd.Series, feature_name: str) -> float:
    for candidate in FEATURE_ALIASES[feature_name]:
        if candidate in row.index:
            numeric_value = pd.to_numeric(row.get(candidate), errors="coerce")
            if pd.notna(numeric_value):
                return float(numeric_value)

    raise RuntimeError(
        f"Nu există o valoare validă pentru câmpul '{feature_name}' în ultima înregistrare din measurements."
    )


def _extract_numeric_column(dataframe: pd.DataFrame, feature_name: str) -> pd.Series:
    for candidate in FEATURE_ALIASES[feature_name]:
        if candidate in dataframe.columns:
            return pd.to_numeric(dataframe[candidate], errors="coerce")

    raise RuntimeError(
        f"Nu există o coloană validă pentru câmpul '{feature_name}' în measurements."
    )


def _compute_zero_streak(values: list[float]) -> int:
    streak = 0
    for value in values:
        if value == 0.0:
            streak += 1
        else:
            break
    return streak


def _has_recent_sudden_zero_drop(values: list[float], recent_zero_streak: int) -> bool:
    if recent_zero_streak < ZERO_WARNING_SUDDEN_DROP_STREAK_THRESHOLD:
        return False
    if len(values) <= recent_zero_streak:
        return False
    # Values are ordered newest->oldest, so this checks a sharp transition from non-zero to consecutive zeros.
    return values[recent_zero_streak] > 0.0


def _build_sensor_warning_map() -> dict[str, str]:
    measurements = get_measurements(limit=ZERO_WARNING_LOOKBACK_ROWS, descending=True, raise_on_error=False)
    if measurements.empty:
        return {}

    warning_map: dict[str, str] = {}

    for feature_name in FEATURE_ALIASES:
        try:
            series = _extract_numeric_column(measurements, feature_name).dropna()
        except RuntimeError:
            continue

        values = [float(value) for value in series.tolist()]
        if not values:
            continue

        zero_count = sum(1 for value in values if value == 0.0)
        zero_ratio = zero_count / len(values)
        recent_zero_streak = _compute_zero_streak(values)
        has_ratio_issue = len(values) >= ZERO_WARNING_MIN_SAMPLES and zero_ratio >= ZERO_WARNING_RATIO_THRESHOLD
        has_streak_issue = recent_zero_streak >= ZERO_WARNING_STREAK_THRESHOLD
        has_sudden_drop = _has_recent_sudden_zero_drop(values, recent_zero_streak)

        if has_ratio_issue or has_streak_issue or has_sudden_drop:
            sudden_drop_note = " Cadere brusca la 0 detectata in ultimele inregistrari." if has_sudden_drop else ""
            warning_map[feature_name] = (
                f"Avertizare senzor: {zero_count}/{len(values)} valori 0 "
                f"({zero_ratio:.0%}), secventa curenta de 0 = {recent_zero_streak}. "
                f"Posibil senzor decuplat sau blocat pe 0.{sudden_drop_note}"
            )

    return warning_map


def _detect_timestamp_column(dataframe: pd.DataFrame) -> str | None:
    lower_map = {column.lower(): column for column in dataframe.columns}
    for candidate in TIMESTAMP_CANDIDATES:
        detected = lower_map.get(candidate.lower())
        if detected is not None:
            return detected
    return None


def _classify_feature_status(feature_name: str, value: float) -> str:
    bands = FEATURE_HUMAN_BANDS.get(feature_name)
    if bands is None:
        return "moderate"

    good_low, good_high, _ = bands["good"]
    moderate_low, moderate_high, _ = bands["moderate"]

    in_good = (good_low is None or value >= good_low) and (good_high is None or value <= good_high)
    if in_good:
        return "good"

    in_moderate = (moderate_low is None or value >= moderate_low) and (moderate_high is None or value <= moderate_high)
    if in_moderate:
        return "moderate"

    return "poor"


def _status_message(feature_name: str, status: str) -> str:
    display_name = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)
    bands = FEATURE_HUMAN_BANDS.get(feature_name, {})
    effect_text = bands.get(status, (None, None, ""))[2] if bands else ""
    if status == "good":
        return f"{display_name}: bun pentru organism. {effect_text}"
    if status == "moderate":
        return f"{display_name}: acceptabil, dar poate cauza disconfort. {effect_text}"
    return f"{display_name}: nivel nefavorabil pentru organism. {effect_text}"


def _format_interval(low: float | None, high: float | None, unit: str) -> str:
    if low is None and high is None:
        return "n/a"
    if low is None:
        return f"<= {high:.1f} {unit}"
    if high is None:
        return f">= {low:.1f} {unit}"
    return f"{low:.1f}-{high:.1f} {unit}"


def _status_reason(feature_name: str, value: float, status: str) -> str:
    bands = FEATURE_HUMAN_BANDS.get(feature_name)
    if bands is None:
        return "Nu exista praguri definite pentru acest parametru."

    unit = FEATURE_UNITS.get(feature_name, "")
    good_low, good_high, _ = bands["good"]
    moderate_low, moderate_high, _ = bands["moderate"]

    if status == "good":
        return (
            f"Valoarea {value:.1f} {unit} este in intervalul bun "
            f"({_format_interval(good_low, good_high, unit)})."
        )

    if status == "moderate":
        if good_high is not None and value > good_high:
            return (
                f"Valoarea {value:.1f} {unit} este peste limita zonei bune ({good_high:.1f} {unit}), "
                f"dar inca in intervalul moderat ({_format_interval(moderate_low, moderate_high, unit)})."
            )
        if good_low is not None and value < good_low:
            return (
                f"Valoarea {value:.1f} {unit} este sub limita zonei bune ({good_low:.1f} {unit}), "
                f"dar inca in intervalul moderat ({_format_interval(moderate_low, moderate_high, unit)})."
            )
        return (
            f"Valoarea {value:.1f} {unit} este in intervalul moderat "
            f"({_format_interval(moderate_low, moderate_high, unit)})."
        )

    if moderate_high is not None and value > moderate_high:
        return (
            f"Valoarea {value:.1f} {unit} depaseste limita superioara moderata ({moderate_high:.1f} {unit}), "
            "deci este clasificata ca poor."
        )
    if moderate_low is not None and value < moderate_low:
        return (
            f"Valoarea {value:.1f} {unit} este sub limita inferioara moderata ({moderate_low:.1f} {unit}), "
            "deci este clasificata ca poor."
        )
    return "Valoarea este in zona poor conform pragurilor definite."


def _build_feature_assessment(
    feature_values: dict[str, float],
    sensor_warning_map: dict[str, str] | None = None,
) -> dict[str, dict[str, str | float | None]]:
    assessment: dict[str, dict[str, str | float | None]] = {}
    sensor_warning_map = sensor_warning_map or {}
    for feature_name, value in feature_values.items():
        status = _classify_feature_status(feature_name, float(value))
        condition_label = _human_condition_label(feature_name, float(value))
        unit = FEATURE_UNITS.get(feature_name, "")
        bands = FEATURE_HUMAN_BANDS.get(feature_name, {})
        good_interval = bands.get("good", (None, None, ""))
        moderate_interval = bands.get("moderate", (None, None, ""))
        assessment[feature_name] = {
            "value": float(value),
            "unit": unit,
            "status": status,
            "condition": condition_label,
            "message": _status_message(feature_name, status),
            "reason": _status_reason(feature_name, float(value), status),
            "good_range": _format_interval(good_interval[0], good_interval[1], unit),
            "moderate_range": _format_interval(moderate_interval[0], moderate_interval[1], unit),
            "sensor_warning": sensor_warning_map.get(feature_name),
        }
    return assessment


def _clamp_feature_value(feature_name: str, value: float) -> float:
    if feature_name == "humidity":
        low, high = HUMIDITY_RANGE
        return float(min(high, max(low, value)))
    return float(max(0.0, value))


def _to_local_timestamp(timestamp: pd.Timestamp) -> pd.Timestamp:
    localized = pd.Timestamp(timestamp)
    if localized.tzinfo is None:
        localized = localized.tz_localize("UTC")
    return localized.tz_convert(LOCAL_TIMEZONE)


def _clamp_forecast_temperature(
    projected_value: float,
    current_value: float,
    forecast_timestamp: pd.Timestamp,
    horizon_hours: int,
) -> float:
    """Keep short-term temperature extrapolations seasonally and physically plausible."""
    local_forecast_timestamp = _to_local_timestamp(forecast_timestamp)
    seasonal_low, seasonal_high = MONTHLY_TEMPERATURE_RANGES[local_forecast_timestamp.month]
    max_change = MAX_TEMPERATURE_CHANGE_PER_HOUR * max(1, horizon_hours)
    short_term_low = current_value - max_change
    short_term_high = current_value + max_change
    low = max(seasonal_low, short_term_low)
    high = min(seasonal_high, short_term_high)
    return float(min(high, max(low, projected_value)))


def _compute_temperature_hour_profile(measurements: pd.DataFrame) -> dict[int, float]:
    """Return local-hour temperature averages from recurring recent observations."""
    timestamp_column = _detect_timestamp_column(measurements)
    if timestamp_column is None:
        return {}

    timestamps = pd.to_datetime(measurements[timestamp_column], errors="coerce", utc=True)
    temperatures = _extract_numeric_column(measurements, "temperature")
    hourly = pd.DataFrame({"timestamp": timestamps, "temperature": temperatures}).dropna()
    if hourly.empty:
        return {}

    hourly["local_hour"] = hourly["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.hour
    profile = hourly.groupby("local_hour")["temperature"].agg(["mean", "count"])
    profile = profile[profile["count"] >= MIN_CALENDAR_PROFILE_SAMPLES]
    return {int(hour): float(row["mean"]) for hour, row in profile.iterrows()}


def _temperature_calendar_adjustment(
    hour_profile: dict[int, float],
    current_timestamp: pd.Timestamp,
    forecast_timestamp: pd.Timestamp,
) -> float:
    """Adjust temperature by the observed local-hour difference between now and the future time."""
    current_hour = _to_local_timestamp(current_timestamp).hour
    forecast_hour = _to_local_timestamp(forecast_timestamp).hour
    current_baseline = hour_profile.get(current_hour)
    forecast_baseline = hour_profile.get(forecast_hour)
    if current_baseline is None or forecast_baseline is None:
        return 0.0
    return float(forecast_baseline - current_baseline)


def _compute_feature_slopes_per_hour(measurements: pd.DataFrame) -> dict[str, float]:
    timestamp_column = _detect_timestamp_column(measurements)
    if timestamp_column is None:
        raise RuntimeError("Nu există coloană de timp pentru calculul prognozei.")

    frame = measurements.copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
    frame = frame.dropna(subset=[timestamp_column])
    if frame.empty:
        raise RuntimeError("Nu există timestamp valid pentru calculul prognozei.")

    frame = frame.sort_values(timestamp_column)
    frame["hour_bucket"] = frame[timestamp_column].dt.floor("h")

    normalized = pd.DataFrame({"hour_bucket": frame["hour_bucket"]})
    for feature_name in FEATURE_ALIASES:
        normalized[feature_name] = _extract_numeric_column(frame, feature_name)

    hourly = normalized.groupby("hour_bucket", as_index=False).mean(numeric_only=True).sort_values("hour_bucket")
    if len(hourly) < 2:
        raise RuntimeError("Sunt necesare cel puțin 2 puncte orare pentru prognoză.")

    x = (hourly["hour_bucket"] - hourly["hour_bucket"].iloc[0]).dt.total_seconds() / 3600.0
    slopes: dict[str, float] = {}

    for feature_name in FEATURE_ALIASES:
        y = pd.to_numeric(hourly[feature_name], errors="coerce")
        valid_mask = y.notna()
        if valid_mask.sum() < 2:
            slopes[feature_name] = 0.0
            continue

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        slope, _ = np.polyfit(x_valid, y_valid, 1)
        slopes[feature_name] = float(slope)

    return slopes


def build_forecast(
    base_feature_values: dict[str, float],
    horizons_hours: list[int],
    lookback_hours: int = 72,
    sensor_warning_map: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    if not horizons_hours:
        return []

    measurements = get_measurements(limit=5000, descending=True, raise_on_error=True)
    if measurements.empty:
        raise RuntimeError("Tabela 'measurements' nu conține date pentru prognoză.")

    timestamp_column = _detect_timestamp_column(measurements)
    if timestamp_column is None:
        raise RuntimeError("Nu există coloană de timp pentru prognoză.")

    frame = measurements.copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
    frame = frame.dropna(subset=[timestamp_column])
    if frame.empty:
        raise RuntimeError("Nu există timestamp valid pentru prognoză.")

    latest_timestamp = frame[timestamp_column].max()
    cutoff = latest_timestamp - pd.Timedelta(hours=max(2, int(lookback_hours)))
    frame = frame[frame[timestamp_column] >= cutoff]
    if len(frame) < 2:
        raise RuntimeError("Nu există suficiente date recente pentru prognoză.")

    slopes = _compute_feature_slopes_per_hour(frame)
    temperature_hour_profile = _compute_temperature_hour_profile(frame)
    sensor_warning_map = sensor_warning_map or _build_sensor_warning_map()
    model = load_model()
    forecast: list[dict[str, object]] = []

    for horizon in sorted(set(int(h) for h in horizons_hours if int(h) > 0)):
        projected_features: dict[str, float] = {}
        forecast_timestamp = latest_timestamp + pd.Timedelta(hours=horizon)
        temperature_calendar_adjustment = _temperature_calendar_adjustment(
            temperature_hour_profile,
            latest_timestamp,
            forecast_timestamp,
        )
        for feature_name, current_value in base_feature_values.items():
            slope_per_hour = slopes.get(feature_name, 0.0)
            projected_raw = float(current_value) + (slope_per_hour * horizon)
            if feature_name == "temperature":
                projected_features[feature_name] = _clamp_forecast_temperature(
                    projected_raw + temperature_calendar_adjustment,
                    current_value=float(current_value),
                    forecast_timestamp=forecast_timestamp,
                    horizon_hours=horizon,
                )
            else:
                projected_features[feature_name] = _clamp_feature_value(feature_name, projected_raw)

        projected_df = pd.DataFrame([projected_features])
        projected_prediction = model.predict(projected_df)[0]
        projected_confidence = float(np.max(model.predict_proba(projected_df)))

        forecast.append(
            {
                "horizon_hours": horizon,
                "forecast_at_local": _to_local_timestamp(forecast_timestamp).isoformat(),
                "prediction": str(projected_prediction),
                "confidence": projected_confidence,
                "input_values": projected_features,
                "feature_assessment": _build_feature_assessment(
                    projected_features,
                    sensor_warning_map=sensor_warning_map,
                ),
                "trend_per_hour": {feature: round(slope, 4) for feature, slope in slopes.items()},
                "temperature_calendar_adjustment": round(temperature_calendar_adjustment, 4),
            }
        )

    return forecast


def _build_hourly_average_features(hours_window: int) -> tuple[pd.DataFrame, dict[str, float]]:
    measurements = get_measurements(limit=5000, descending=True, raise_on_error=True)
    if measurements.empty:
        raise RuntimeError("Tabela 'measurements' nu conține date pentru predicție.")

    timestamp_column = _detect_timestamp_column(measurements)
    if timestamp_column is None:
        raise RuntimeError("Nu există coloană de timp pentru agregarea pe oră.")

    window_frame = measurements.copy()
    window_frame[timestamp_column] = pd.to_datetime(window_frame[timestamp_column], errors="coerce", utc=True)
    window_frame = window_frame.dropna(subset=[timestamp_column])
    if window_frame.empty:
        raise RuntimeError("Nu există timestamp valid pentru agregarea pe oră.")

    latest_timestamp = window_frame[timestamp_column].max()
    cutoff = latest_timestamp - pd.Timedelta(hours=int(hours_window))
    window_frame = window_frame[window_frame[timestamp_column] >= cutoff]
    if window_frame.empty:
        raise RuntimeError("Nu există date în intervalul orar selectat pentru predicție.")

    feature_values = {}
    for feature_name in FEATURE_ALIASES:
        numeric_values = _extract_numeric_column(window_frame, feature_name)
        mean_value = numeric_values.mean(skipna=True)
        if pd.isna(mean_value):
            raise RuntimeError(
                f"Nu există valori numerice pentru câmpul '{feature_name}' în intervalul selectat."
            )
        feature_values[feature_name] = float(mean_value)

    input_df = pd.DataFrame([feature_values])
    return input_df, feature_values


def _load_latest_measurement_features() -> tuple[pd.DataFrame, dict[str, float]]:
    measurements = get_measurements(limit=1, descending=True, raise_on_error=True)
    if measurements.empty:
        raise RuntimeError("Tabela 'measurements' nu conține date pentru predicție.")

    latest = measurements.iloc[0]
    feature_values = {
        feature_name: _extract_measurement_feature(latest, feature_name)
        for feature_name in FEATURE_ALIASES
    }
    input_df = pd.DataFrame([feature_values])
    return input_df, feature_values


def predict_air_quality(use_hourly_average: bool = False, aggregation_hours: int = 1):
    if use_hourly_average:
        input_df, feature_values = _build_hourly_average_features(hours_window=max(1, int(aggregation_hours)))
    else:
        input_df, feature_values = _load_latest_measurement_features()

    model = load_model()
    prediction = model.predict(input_df)[0]
    confidence = float(np.max(model.predict_proba(input_df)))
    sensor_warning_map = _build_sensor_warning_map()
    feature_assessment = _build_feature_assessment(feature_values, sensor_warning_map=sensor_warning_map)
    return prediction, confidence, feature_values, feature_assessment
