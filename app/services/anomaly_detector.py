import pandas as pd

from app.core import config
from app.core.database import get_measurements
from app.models.train_model import load_model


FEATURE_ALIASES = {
    "temperature": ["temperature", "temperatura", "temp"],
    "humidity": ["humidity", "umiditate"],
    "pm25": ["pm25", "pm2_5", "pm2.5"],
    "pm10": ["pm10"],
    "co2": ["co2", "co_2"],
}
ANALYSIS_BASELINE_ROWS = 500
ZERO_WARNING_MIN_SAMPLES = 24
ZERO_WARNING_RATIO_THRESHOLD = 0.6
ZERO_WARNING_STREAK_THRESHOLD = 8
ZERO_WARNING_SUDDEN_DROP_STREAK_THRESHOLD = 3


def _extract_measurement_feature(row: pd.Series, feature_name: str) -> float:
    for candidate in FEATURE_ALIASES[feature_name]:
        if candidate in row.index:
            numeric_value = pd.to_numeric(row.get(candidate), errors="coerce")
            if pd.notna(numeric_value):
                return float(numeric_value)

    raise RuntimeError(
        f"Nu există o valoare validă pentru câmpul '{feature_name}' în ultima înregistrare din measurements."
    )


def _load_latest_measurement_features() -> tuple[pd.DataFrame, dict[str, float]]:
    measurements = get_measurements(limit=1, descending=True, raise_on_error=True)
    if measurements.empty:
        raise RuntimeError("Tabela 'measurements' nu conține date pentru detecția anomaliilor.")

    latest = measurements.iloc[0]
    feature_values = {
        feature_name: _extract_measurement_feature(latest, feature_name)
        for feature_name in FEATURE_ALIASES
    }
    input_df = pd.DataFrame([feature_values])
    return input_df, feature_values


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


def _build_feature_analysis(
    current_values: dict[str, float],
) -> tuple[list[dict[str, float | str | bool]], list[str], list[str]]:
    baseline_df = get_measurements(limit=ANALYSIS_BASELINE_ROWS, descending=True, raise_on_error=False)
    if baseline_df.empty:
        return [], [], []

    analysis_rows: list[dict[str, float | str | bool]] = []
    anomalous_features: list[str] = []
    sensor_health_warnings: list[str] = []

    for feature_name, candidates in FEATURE_ALIASES.items():
        source_column = next((column for column in candidates if column in baseline_df.columns), None)
        if source_column is None:
            continue

        numeric_series = pd.to_numeric(baseline_df[source_column], errors="coerce").dropna()
        if numeric_series.empty:
            continue

        q1 = float(numeric_series.quantile(0.25))
        q3 = float(numeric_series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        current_value = float(current_values[feature_name])
        is_outlier = current_value < lower_bound or current_value > upper_bound
        values = [float(value) for value in numeric_series.tolist()]
        zero_count = sum(1 for value in values if value == 0.0)
        zero_ratio = zero_count / len(values)
        recent_zero_streak = _compute_zero_streak(values)
        sensor_warning = ""
        has_ratio_issue = len(values) >= ZERO_WARNING_MIN_SAMPLES and zero_ratio >= ZERO_WARNING_RATIO_THRESHOLD
        has_streak_issue = recent_zero_streak >= ZERO_WARNING_STREAK_THRESHOLD
        has_sudden_drop = _has_recent_sudden_zero_drop(values, recent_zero_streak)

        if has_ratio_issue or has_streak_issue or has_sudden_drop:
            sudden_drop_note = " Cadere brusca la 0 detectata in ultimele inregistrari." if has_sudden_drop else ""
            sensor_warning = (
                f"Avertizare senzor pentru {feature_name}: {zero_count}/{len(values)} valori 0 "
                f"({zero_ratio:.0%}), secventa curenta de 0 = {recent_zero_streak}. "
                f"Posibil senzor decuplat sau blocat pe 0.{sudden_drop_note}"
            )
            sensor_health_warnings.append(sensor_warning)

        if is_outlier:
            anomalous_features.append(feature_name)

        analysis_rows.append(
            {
                "feature": feature_name,
                "value": current_value,
                "q1": q1,
                "q3": q3,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "is_outlier": is_outlier,
                "zero_ratio": zero_ratio,
                "recent_zero_streak": recent_zero_streak,
                "sensor_warning": sensor_warning,
            }
        )

    return analysis_rows, anomalous_features, sensor_health_warnings


def detect_anomaly():
    input_df, feature_values = _load_latest_measurement_features()
    model = load_model(config.IF_MODEL_PATH)
    feature_analysis, anomalous_features, sensor_health_warnings = _build_feature_analysis(feature_values)

    prediction = int(model.predict(input_df)[0])
    score = float(model.decision_function(input_df)[0])
    is_anomaly = prediction == -1

    return {
        "is_anomaly": is_anomaly,
        "prediction": prediction,
        "score": score,
        "label": "anomaly" if is_anomaly else "normal",
        "input_values": feature_values,
        "anomalous_features": anomalous_features,
        "feature_analysis": feature_analysis,
        "sensor_health_warnings": sensor_health_warnings,
    }
