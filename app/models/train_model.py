import joblib
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.core.config import IF_MODEL_PATH, MODELS_DIR, RF_MODEL_PATH, SVM_MODEL_PATH, XGBOOST_MODEL_PATH
from app.core.database import get_measurements


FEATURE_COLUMNS = ["temperature", "humidity", "pm25", "pm10", "co2"]
TARGET_COLUMN = "quality_label"
LATEST_TRAINING_ROWS = 1000
EVOLUTION_STEPS = [10, 20, 40, 60, 80, 100]
TEST_SIZE = 0.2
TIMESTAMP_CANDIDATES = ["created_at", "timestamp", "time", "recorded_at"]
TIME_AGGREGATION_GRANULARITIES = {"hour", "minute"}


VOC_FEATURE_NAME = "voc"
DATABASE_LABEL_SOURCE = "database_quality_label"
SYNTHETIC_LABEL_SOURCE = "derived_from_features"
QUANTILE_LABEL_SOURCE = "derived_quantile_labels"
LABEL_SOURCE_COLUMN_CANDIDATES = ["quality_label_source", "label_source", "quality_source"]
INDEPENDENT_LABEL_SOURCES = {
    "manual",
    "expert_review",
    "external_aqi_standard",
    "lab_reference",
    "independent_sensor_fusion",
}
HARD_POOR_LIMITS = {
    "pm25": 55.0,
    "pm10": 155.0,
    "co2": 2000.0,
    "temperature_low": 10.0,
    "temperature_high": 35.0,
    "humidity_low": 15.0,
    "humidity_high": 85.0,
    "voc": 1200.0,
}


def _normalize_measurement_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    aliases = {
        "temperature": ["temperature", "temperatura"],
        "humidity": ["humidity", "umiditate"],
        "pm25": ["pm25"],
        "pm10": ["pm10"],
        "co2": ["co2"],
        "voc": ["voc", "tvoc"],
    }

    for feature_name, candidate_columns in aliases.items():
        source_column = next((column for column in candidate_columns if column in normalized.columns), None)
        if source_column is not None:
            normalized[feature_name] = pd.to_numeric(normalized[source_column], errors="coerce")

    return normalized


def _detect_timestamp_column(dataframe: pd.DataFrame) -> str | None:
    lower_map = {column.lower(): column for column in dataframe.columns}
    for candidate in TIMESTAMP_CANDIDATES:
        detected = lower_map.get(candidate.lower())
        if detected is not None:
            return detected
    return None


def _find_time_column_for_report(dataframe: pd.DataFrame) -> str | None:
    if "hour_bucket" in dataframe.columns:
        return "hour_bucket"
    return _detect_timestamp_column(dataframe)


def _isoformat_or_none(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return timestamp.isoformat()


def _build_dataset_info(
    training_frame: pd.DataFrame,
    labels: pd.Series | None,
    include_class_distribution: bool,
) -> dict:
    time_column = _find_time_column_for_report(training_frame)
    time_start = None
    time_end = None
    if time_column is not None:
        parsed_time = pd.to_datetime(training_frame[time_column], errors="coerce", utc=True)
        if parsed_time.notna().any():
            time_start = _isoformat_or_none(parsed_time.min())
            time_end = _isoformat_or_none(parsed_time.max())

    device_count = 0
    if "device_identifier" in training_frame.columns:
        device_count = int(
            training_frame["device_identifier"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .nunique()
        )

    class_distribution = None
    if include_class_distribution and labels is not None:
        counts = labels.value_counts().to_dict()
        class_distribution = {str(label): int(count) for label, count in counts.items()}

    return {
        "total_measurements": int(len(training_frame)),
        "time_range": {
            "start": time_start,
            "end": time_end,
        },
        "device_count": device_count,
        "class_distribution": class_distribution,
    }


def _mode_label(series: pd.Series) -> str | None:
    non_null = series.dropna()
    if non_null.empty:
        return None
    modes = non_null.mode(dropna=True)
    if modes.empty:
        return None
    return str(modes.iloc[0])


def _resolve_time_aggregation(
    aggregation_hours: int,
    aggregation_minutes: int | None,
) -> tuple[str, int]:
    if aggregation_minutes is not None:
        if int(aggregation_minutes) < 1:
            raise RuntimeError("aggregation_minutes trebuie sa fie >= 1.")
        if int(aggregation_minutes) > 60:
            raise RuntimeError("aggregation_minutes trebuie sa fie <= 60. Pentru intervale mai mari, foloseste agregarea pe ore.")
        return "minute", int(aggregation_minutes)
    if int(aggregation_hours) < 1:
        raise RuntimeError("aggregation_hours trebuie sa fie >= 1.")

    # Hour selection is interpreted as a minute-bucket window.
    # Example: 1h -> 60 buckets, 2h -> 120 buckets.
    return "minute", int(aggregation_hours) * 60


def _aggregate_time_training_rows(
    dataframe: pd.DataFrame,
    aggregation_value: int,
    aggregation_granularity: str,
) -> pd.DataFrame:
    if aggregation_granularity not in TIME_AGGREGATION_GRANULARITIES:
        raise RuntimeError("Granularitate invalida pentru agregare. Foloseste 'hour' sau 'minute'.")

    timestamp_column = _detect_timestamp_column(dataframe)
    if timestamp_column is None:
        raise RuntimeError("Nu există coloană de timp pentru agregarea temporală la antrenare.")

    working_frame = dataframe.copy()
    working_frame[timestamp_column] = pd.to_datetime(working_frame[timestamp_column], errors="coerce", utc=True)
    working_frame = working_frame.dropna(subset=[timestamp_column])
    if working_frame.empty:
        raise RuntimeError("Nu există timestamp valid pentru agregarea temporală la antrenare.")

    if aggregation_granularity == "minute":
        resolved_minutes = max(1, int(aggregation_value))
        working_frame["minute_bucket"] = working_frame[timestamp_column].dt.floor("min")
        feature_by_minute = (
            working_frame
            .groupby("minute_bucket", dropna=False)[FEATURE_COLUMNS]
            .mean()
            .sort_index()
        )
        if feature_by_minute.empty:
            raise RuntimeError("Nu există minute valide pentru agregarea temporală la antrenare.")

        latest_bucket = feature_by_minute.index.max()
        target_index = pd.date_range(
            end=latest_bucket,
            periods=resolved_minutes,
            freq="min",
            tz=latest_bucket.tz,
        )

        feature_by_minute = feature_by_minute.reindex(target_index).ffill().bfill()
        aggregated = (
            feature_by_minute
            .reset_index()
            .rename(columns={"index": "minute_bucket"})
        )

        if TARGET_COLUMN in working_frame.columns:
            label_by_minute = (
                working_frame
                .groupby("minute_bucket", dropna=False)[TARGET_COLUMN]
                .agg(_mode_label)
                .sort_index()
                .reindex(target_index)
                .ffill()
                .bfill()
            )
            aggregated[TARGET_COLUMN] = label_by_minute.values

        return aggregated

    latest_timestamp = working_frame[timestamp_column].max()
    cutoff = latest_timestamp - pd.Timedelta(hours=max(1, int(aggregation_value)))
    working_frame = working_frame[working_frame[timestamp_column] >= cutoff]
    if working_frame.empty:
        raise RuntimeError("Nu există date în intervalul temporal selectat pentru antrenare.")
    working_frame["hour_bucket"] = working_frame[timestamp_column].dt.floor("h")

    group_columns = ["hour_bucket"]

    grouped = working_frame.groupby(group_columns, dropna=False)

    aggregated = grouped[FEATURE_COLUMNS].mean().reset_index(drop=False)
    if TARGET_COLUMN in working_frame.columns:
        label_frame = grouped[TARGET_COLUMN].agg(_mode_label).reset_index(drop=False)
        aggregated = aggregated.merge(label_frame, on=[column for column in label_frame.columns if column != TARGET_COLUMN], how="left")

    return aggregated


def _metric_score(value: float, good_max: float, moderate_max: float) -> int:
    if value <= good_max:
        return 0
    if value <= moderate_max:
        return 1
    return 2


def _range_score(value: float, good_low: float, good_high: float, moderate_low: float, moderate_high: float) -> int:
    if good_low <= value <= good_high:
        return 0
    if moderate_low <= value <= moderate_high:
        return 1
    return 2


def _label_from_measurement_row(row: pd.Series) -> str:
    scores: dict[str, int] = {
        "pm25": _metric_score(float(row["pm25"]), good_max=15.0, moderate_max=35.0),
        "pm10": _metric_score(float(row["pm10"]), good_max=45.0, moderate_max=90.0),
        "co2": _metric_score(float(row["co2"]), good_max=800.0, moderate_max=1200.0),
        "temperature": _range_score(float(row["temperature"]), good_low=20.0, good_high=26.0, moderate_low=17.0, moderate_high=30.0),
        "humidity": _range_score(float(row["humidity"]), good_low=30.0, good_high=60.0, moderate_low=20.0, moderate_high=70.0),
    }

    voc_value = row.get(VOC_FEATURE_NAME, pd.NA)
    if pd.notna(voc_value):
        scores[VOC_FEATURE_NAME] = _metric_score(float(voc_value), good_max=250.0, moderate_max=600.0)

    hard_poor = (
        float(row["pm25"]) >= HARD_POOR_LIMITS["pm25"]
        or float(row["pm10"]) >= HARD_POOR_LIMITS["pm10"]
        or float(row["co2"]) >= HARD_POOR_LIMITS["co2"]
        or float(row["temperature"]) < HARD_POOR_LIMITS["temperature_low"]
        or float(row["temperature"]) > HARD_POOR_LIMITS["temperature_high"]
        or float(row["humidity"]) < HARD_POOR_LIMITS["humidity_low"]
        or float(row["humidity"]) > HARD_POOR_LIMITS["humidity_high"]
        or (VOC_FEATURE_NAME in scores and float(row[VOC_FEATURE_NAME]) >= HARD_POOR_LIMITS["voc"])
    )

    if hard_poor:
        return "poor"

    weights = {
        "pm25": 0.30,
        "pm10": 0.20,
        "co2": 0.20,
        "temperature": 0.10,
        "humidity": 0.10,
        VOC_FEATURE_NAME: 0.10,
    }

    weighted_sum = 0.0
    total_weight = 0.0
    for feature_name, score in scores.items():
        weight = weights.get(feature_name, 0.0)
        weighted_sum += score * weight
        total_weight += weight

    risk_index = weighted_sum / total_weight if total_weight > 0 else 0.0
    poor_count = sum(1 for score in scores.values() if score == 2)
    moderate_count = sum(1 for score in scores.values() if score == 1)

    if poor_count >= 2 or risk_index >= 1.05:
        return "poor"
    if poor_count >= 1 or moderate_count >= 2 or risk_index >= 0.45:
        return "moderate"
    return "good"


def _coerce_existing_quality_labels(raw_labels: pd.Series) -> pd.Series:
    normalized = raw_labels.astype(str).str.strip().str.lower()
    valid_labels = {"good", "moderate", "poor"}
    return normalized.where(normalized.isin(valid_labels), pd.NA)


def _find_label_source_column(dataframe: pd.DataFrame) -> str | None:
    lower_map = {column.lower(): column for column in dataframe.columns}
    for candidate in LABEL_SOURCE_COLUMN_CANDIDATES:
        detected = lower_map.get(candidate.lower())
        if detected is not None:
            return detected
    return None


def _build_supervised_labels_requirements_message(reason: str) -> str:
    return (
        f"{reason} "
        "Pentru antrenare Random Forest/SVM/XGBoost fara data leakage, tabela 'measurements' trebuie sa includa: "
        "(1) quality_label cu valori din {good, moderate, poor}; "
        "(2) quality_label_source cu valori independente de feature-urile modelului "
        "(ex: manual, expert_review, external_aqi_standard, lab_reference, independent_sensor_fusion)."
    )


def _derive_quantile_labels(frame: pd.DataFrame) -> pd.Series:
    risk_index = (
        frame["pm25"].astype(float) * 0.30
        + frame["pm10"].astype(float) * 0.20
        + frame["co2"].astype(float) * 0.20
        + frame["temperature"].astype(float) * 0.15
        + frame["humidity"].astype(float) * 0.15
    )

    rank = risk_index.rank(method="first", pct=True)
    derived = pd.Series("moderate", index=frame.index, dtype="object")
    derived[rank <= (1 / 3)] = "good"
    derived[rank >= (2 / 3)] = "poor"

    if derived.nunique(dropna=True) < 2 and len(derived) >= 2:
        split_rank = frame["pm25"].astype(float).rank(method="first", pct=True)
        derived = pd.Series("moderate", index=frame.index, dtype="object")
        derived[split_rank <= 0.5] = "good"

    return derived


def _load_training_frame(
    use_hourly_aggregation: bool = False,
    aggregation_hours: int = 24,
    aggregation_minutes: int | None = None,
    row_limit: int = LATEST_TRAINING_ROWS,
    require_labels: bool = True,
    allow_derived_label_fallback: bool = False,
) -> pd.DataFrame:
    fetch_limit = None if use_hourly_aggregation else int(row_limit)
    dataframe = get_measurements(limit=fetch_limit, descending=True, raise_on_error=True)
    normalized = _normalize_measurement_frame(dataframe)

    if normalized.empty:
        raise RuntimeError("Tabela 'measurements' nu conține date pentru antrenare.")

    missing_features = [column for column in FEATURE_COLUMNS if column not in normalized.columns]
    if missing_features:
        raise RuntimeError(
            "Lipsesc coloane necesare pentru antrenare din tabela 'measurements': "
            + ", ".join(missing_features)
        )

    training_frame = normalized.dropna(subset=FEATURE_COLUMNS).copy()
    if training_frame.empty:
        raise RuntimeError("Nu există rânduri valide în tabela 'measurements' pentru antrenare.")

    label_source = DATABASE_LABEL_SOURCE
    if require_labels:
        label_frame: pd.DataFrame | None = None
        label_error_message: str | None = None

        if TARGET_COLUMN not in training_frame.columns:
            label_error_message = _build_supervised_labels_requirements_message(
                "Lipsește coloana 'quality_label' în tabela 'measurements'."
            )
        else:
            label_source_column = _find_label_source_column(training_frame)
            if label_source_column is None:
                label_error_message = _build_supervised_labels_requirements_message(
                    "Lipsește coloana 'quality_label_source' în tabela 'measurements'."
                )
            else:
                db_labels = _coerce_existing_quality_labels(training_frame[TARGET_COLUMN])
                source_values = training_frame[label_source_column].astype(str).str.strip().str.lower()
                independent_source_mask = source_values.isin(INDEPENDENT_LABEL_SOURCES)
                valid_rows_mask = db_labels.notna() & independent_source_mask
                valid_rows = int(valid_rows_mask.sum())

                if valid_rows < 10:
                    total_rows = int(len(training_frame))
                    label_error_message = _build_supervised_labels_requirements_message(
                        "Date insuficiente pentru etichete independente: "
                        f"{valid_rows} rânduri valide din {total_rows} disponibile."
                    )
                else:
                    label_frame = training_frame.loc[valid_rows_mask].copy()
                    label_frame[TARGET_COLUMN] = db_labels.loc[valid_rows_mask]
                    if label_frame[TARGET_COLUMN].nunique() < 2:
                        label_error_message = _build_supervised_labels_requirements_message(
                            "Sunt necesare cel puțin două clase distincte în quality_label."
                        )

        if label_frame is not None:
            training_frame = label_frame
        elif allow_derived_label_fallback:
            training_frame[TARGET_COLUMN] = training_frame.apply(_label_from_measurement_row, axis=1)
            training_frame = training_frame.dropna(subset=[TARGET_COLUMN]).copy()
            label_source = SYNTHETIC_LABEL_SOURCE
        else:
            raise RuntimeError(label_error_message or "Etichetele de antrenare nu sunt valide.")

    if use_hourly_aggregation:
        source_rows_before_aggregation = int(len(training_frame))
        aggregation_granularity, aggregation_value = _resolve_time_aggregation(
            aggregation_hours=aggregation_hours,
            aggregation_minutes=aggregation_minutes,
        )
        training_frame = _aggregate_time_training_rows(
            training_frame,
            aggregation_value=aggregation_value,
            aggregation_granularity=aggregation_granularity,
        )
        training_frame = training_frame.dropna(subset=FEATURE_COLUMNS).copy()
        if training_frame.empty:
            raise RuntimeError("Nu există rânduri agregate valide pentru antrenare.")
        training_frame.attrs["source_rows_before_aggregation"] = source_rows_before_aggregation
        training_frame.attrs["aggregation_granularity"] = aggregation_granularity
        training_frame.attrs["aggregation_value"] = int(aggregation_value)
    else:
        training_frame.attrs["aggregation_granularity"] = None
        training_frame.attrs["aggregation_value"] = None

    if require_labels:
        training_frame = training_frame.dropna(subset=[TARGET_COLUMN]).copy()
        if training_frame[TARGET_COLUMN].nunique() < 2:
            if allow_derived_label_fallback:
                training_frame[TARGET_COLUMN] = _derive_quantile_labels(training_frame)
                label_source = QUANTILE_LABEL_SOURCE
            else:
                raise RuntimeError(
                    "După agregarea orară au rămas mai puțin de două clase în quality_label. "
                    "Reduce intervalul orar sau crește volumul de date etichetate independent."
                )

        if training_frame[TARGET_COLUMN].nunique() < 2:
            raise RuntimeError("Sunt necesare cel puțin două clase pentru antrenarea modelului.")

    training_frame.attrs["label_source"] = label_source
    training_frame.attrs["hourly_aggregation"] = bool(use_hourly_aggregation)
    training_frame.attrs["aggregation_hours"] = int(aggregation_hours)
    training_frame.attrs["aggregation_minutes"] = int(aggregation_minutes) if aggregation_minutes is not None else None
    training_frame.attrs["effective_row_limit"] = None if use_hourly_aggregation else int(row_limit)

    return training_frame


def load_training_data(
    use_hourly_aggregation: bool = False,
    aggregation_hours: int = 24,
    aggregation_minutes: int | None = None,
    row_limit: int = LATEST_TRAINING_ROWS,
    allow_derived_label_fallback: bool = False,
):
    training_frame = _load_training_frame(
        use_hourly_aggregation=use_hourly_aggregation,
        aggregation_hours=aggregation_hours,
        aggregation_minutes=aggregation_minutes,
        row_limit=row_limit,
        require_labels=True,
        allow_derived_label_fallback=allow_derived_label_fallback,
    )
    X = training_frame[FEATURE_COLUMNS]
    y = training_frame[TARGET_COLUMN]
    return X, y


def _load_supervised_training_parts(
    use_hourly_aggregation: bool = False,
    aggregation_hours: int = 24,
    aggregation_minutes: int | None = None,
    row_limit: int = LATEST_TRAINING_ROWS,
    allow_derived_label_fallback: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, str, bool, str | None, int | None, int | None, int | None]:
    training_frame = _load_training_frame(
        use_hourly_aggregation=use_hourly_aggregation,
        aggregation_hours=aggregation_hours,
        aggregation_minutes=aggregation_minutes,
        row_limit=row_limit,
        require_labels=True,
        allow_derived_label_fallback=allow_derived_label_fallback,
    )
    label_source = str(training_frame.attrs.get("label_source", DATABASE_LABEL_SOURCE))
    hourly_aggregation = bool(training_frame.attrs.get("hourly_aggregation", False))
    aggregation_granularity = training_frame.attrs.get("aggregation_granularity")
    aggregation_value = training_frame.attrs.get("aggregation_value")
    effective_row_limit = training_frame.attrs.get("effective_row_limit")
    source_rows_before_aggregation = training_frame.attrs.get("source_rows_before_aggregation")
    X = training_frame[FEATURE_COLUMNS]
    y = training_frame[TARGET_COLUMN]
    return (
        X,
        y,
        training_frame,
        label_source,
        hourly_aggregation,
        aggregation_granularity,
        int(aggregation_value) if aggregation_value is not None else None,
        effective_row_limit,
        source_rows_before_aggregation,
    )


def _build_training_summary(
    y: pd.Series,
    label_source: str,
    hourly_aggregation: bool,
    aggregation_hours: int,
    row_limit: int | None,
    aggregation_granularity: str | None = None,
    aggregation_value: int | None = None,
    source_rows_before_aggregation: int | None = None,
) -> dict:
    class_counts = y.value_counts().to_dict()
    return {
        "rows_requested": int(row_limit) if row_limit is not None else None,
        "source_rows_before_aggregation": (
            int(source_rows_before_aggregation) if source_rows_before_aggregation is not None else None
        ),
        "rows_used": int(len(y)),
        "label_source": label_source,
        "hourly_aggregation": hourly_aggregation,
        "aggregation_hours": int(aggregation_hours),
        "aggregation_granularity": aggregation_granularity,
        "aggregation_value": int(aggregation_value) if aggregation_value is not None else None,
        "class_distribution": {str(label): int(count) for label, count in class_counts.items()},
    }


def _build_classifier_evaluation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict | None:
    labels = sorted(y.unique().tolist())

    if len(X) < 10:
        return None

    # Avoid identical feature+label rows crossing between train/test.
    group_frame = X.copy()
    group_frame["__target"] = y.astype(str)
    row_groups = group_frame.astype(str).agg("|".join, axis=1)

    if row_groups.nunique() < 2:
        return None

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    try:
        train_idx, test_idx = next(splitter.split(X, y, groups=row_groups))
    except ValueError:
        return None

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    if len(X_train) == 0 or len(X_test) == 0:
        return None

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=42,
                stratify=y,
            )
        except ValueError:
            return None

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    evaluation = {
        "train_examples": int(len(X_train)),
        "test_examples": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": {
            "labels": [str(label) for label in labels],
            "matrix": matrix.tolist(),
        },
        "classification_report": report_dict,
        "evaluation_scope": "holdout_test",
    }

    return evaluation


def _train_random_forest_with_evolution(X: pd.DataFrame, y: pd.Series) -> tuple[RandomForestClassifier, list[dict]]:
    model = RandomForestClassifier(
        n_estimators=EVOLUTION_STEPS[0],
        random_state=42,
        warm_start=True,
        oob_score=True,
        bootstrap=True,
    )
    evolution: list[dict] = []

    for step in EVOLUTION_STEPS:
        model.set_params(n_estimators=step)
        model.fit(X, y)
        # OOB is not informative on extremely small samples.
        if hasattr(model, "oob_score_") and len(X) >= 10:
            oob_score = float(model.oob_score_)
        else:
            oob_score = None
        evolution.append(
            {
                "step": int(step),
                "oob_score": oob_score,
            }
        )

    model.set_params(warm_start=False)
    return model, evolution


def _train_isolation_forest_with_evolution(X: pd.DataFrame) -> tuple[IsolationForest, list[dict]]:
    model = IsolationForest(
        n_estimators=EVOLUTION_STEPS[0],
        contamination=0.2,
        random_state=42,
        warm_start=True,
    )
    evolution: list[dict] = []

    for step in EVOLUTION_STEPS:
        model.set_params(n_estimators=step)
        model.fit(X)
        decision_scores = model.decision_function(X)
        evolution.append(
            {
                "step": int(step),
                "mean_decision_score": float(decision_scores.mean()),
            }
        )

    model.set_params(warm_start=False)
    return model, evolution


def _fit_and_save(model, target_path: Path):
    X, y = load_training_data()

    model.fit(X, y)
    print(
        f"Trained {model.__class__.__name__} on {len(X)} latest rows from Supabase measurements "
        f"(limit={LATEST_TRAINING_ROWS})."
    )

    joblib.dump(model, target_path)
    print(f"Model saved to {target_path}")
    return model


def train_and_save_model(
    model_path: str | Path | None = None,
    return_report: bool = False,
    use_hourly_aggregation: bool = False,
    aggregation_hours: int = 24,
    aggregation_minutes: int | None = None,
    row_limit: int = LATEST_TRAINING_ROWS,
    allow_derived_label_fallback: bool = False,
):
    return train_and_save_random_forest(
        model_path=model_path,
        return_report=return_report,
        use_hourly_aggregation=use_hourly_aggregation,
        aggregation_hours=aggregation_hours,
        aggregation_minutes=aggregation_minutes,
        row_limit=row_limit,
        allow_derived_label_fallback=allow_derived_label_fallback,
    )


def train_and_save_random_forest(
    model_path: str | Path | None = None,
    return_report: bool = False,
    use_hourly_aggregation: bool = False,
    aggregation_hours: int = 24,
    aggregation_minutes: int | None = None,
    row_limit: int = LATEST_TRAINING_ROWS,
    allow_derived_label_fallback: bool = False,
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    target_path = Path(model_path) if model_path is not None else RF_MODEL_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    (
        X,
        y,
        training_frame,
        label_source,
        hourly_aggregation,
        aggregation_granularity,
        aggregation_value,
        requested_rows,
        source_rows_before_aggregation,
    ) = _load_supervised_training_parts(
        use_hourly_aggregation=use_hourly_aggregation,
        aggregation_hours=aggregation_hours,
        aggregation_minutes=aggregation_minutes,
        row_limit=row_limit,
        allow_derived_label_fallback=allow_derived_label_fallback,
    )
    evaluation_model = RandomForestClassifier(n_estimators=100, random_state=42)
    evaluation = None
    if label_source == DATABASE_LABEL_SOURCE:
        evaluation = _build_classifier_evaluation(evaluation_model, X, y)

    model, evolution = _train_random_forest_with_evolution(X, y)
    joblib.dump(model, target_path)

    dataset_info = _build_dataset_info(
        training_frame=training_frame,
        labels=y,
        include_class_distribution=True,
    )
    model_info = {
        "name": "Random Forest Classifier",
        "n_estimators": int(model.n_estimators),
        "last_trained_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(target_path),
    }

    evaluation_note = None
    if label_source != DATABASE_LABEL_SOURCE:
        evaluation_note = (
            "Antrenarea a folosit etichete derivate din feature-uri (mod fallback). "
            "Acest lucru introduce data leakage, deci metricele Accuracy/Precision/Recall/F1 "
            "nu sunt relevante pentru performanța reală."
        )
    elif evaluation is None:
        evaluation_note = (
            "Metricele clasice nu au fost calculate deoarece setul etichetat independent "
            "este prea mic pentru o evaluare holdout stabilă."
        )

    report = {
        "model_type": "random_forest",
        "dataset_info": dataset_info,
        "model_info": model_info,
        "summary": _build_training_summary(
            y,
            label_source=label_source,
            hourly_aggregation=hourly_aggregation,
            aggregation_hours=aggregation_hours,
            aggregation_granularity=aggregation_granularity,
            aggregation_value=aggregation_value,
            row_limit=requested_rows,
            source_rows_before_aggregation=source_rows_before_aggregation,
        ),
        "technical_details": {
            "evolution": evolution,
            "feature_importances": {
                FEATURE_COLUMNS[i]: float(model.feature_importances_[i])
                for i in range(len(FEATURE_COLUMNS))
            },
        },
        "evaluation_note": evaluation_note,
        "evaluation": evaluation,
    }

    print(
        f"Trained RandomForestClassifier on {len(X)} rows from Supabase measurements "
        f"(limit={requested_rows if requested_rows is not None else 'hours-window'})."
    )
    print(f"Model saved to {target_path}")
    if return_report:
        return model, report
    return model


def train_and_save_svm(
    model_path: str | Path | None = None,
    return_report: bool = False,
    use_hourly_aggregation: bool = False,
    aggregation_hours: int = 24,
    aggregation_minutes: int | None = None,
    row_limit: int = LATEST_TRAINING_ROWS,
    allow_derived_label_fallback: bool = False,
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    target_path = Path(model_path) if model_path is not None else SVM_MODEL_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # SVM benefits from standardized features.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=42)),
        ]
    )
    (
        X,
        y,
        training_frame,
        label_source,
        hourly_aggregation,
        aggregation_granularity,
        aggregation_value,
        requested_rows,
        source_rows_before_aggregation,
    ) = _load_supervised_training_parts(
        use_hourly_aggregation=use_hourly_aggregation,
        aggregation_hours=aggregation_hours,
        aggregation_minutes=aggregation_minutes,
        row_limit=row_limit,
        allow_derived_label_fallback=allow_derived_label_fallback,
    )
    evaluation = None
    if label_source == DATABASE_LABEL_SOURCE:
        evaluation = _build_classifier_evaluation(model, X, y)
    model.fit(X, y)
    trained_model = model
    joblib.dump(trained_model, target_path)

    if return_report:
        dataset_info = _build_dataset_info(
            training_frame=training_frame,
            labels=y,
            include_class_distribution=True,
        )
        model_info = {
            "name": "SVM Classifier",
            "kernel": "rbf",
            "last_trained_at": datetime.now(timezone.utc).isoformat(),
            "model_path": str(target_path),
        }
        
        evaluation_note = None
        if label_source != DATABASE_LABEL_SOURCE:
            evaluation_note = (
                "Antrenarea a folosit etichete derivate din feature-uri (mod fallback). "
                "Acest lucru introduce data leakage, deci metricele Accuracy/Precision/Recall/F1 "
                "nu sunt relevante pentru performanța reală."
            )
        elif evaluation is None:
            evaluation_note = (
                "Metricile clasice nu au fost calculate deoarece setul etichetat independent "
                "este prea mic pentru o evaluare holdout stabilă."
            )
        
        report = {
            "model_type": "svm",
            "dataset_info": dataset_info,
            "model_info": model_info,
            "summary": _build_training_summary(
                y,
                label_source=label_source,
                hourly_aggregation=hourly_aggregation,
                aggregation_hours=aggregation_hours,
                aggregation_granularity=aggregation_granularity,
                aggregation_value=aggregation_value,
                row_limit=requested_rows,
                source_rows_before_aggregation=source_rows_before_aggregation,
            ),
            "technical_details": {
                "feature_importances": None,  # SVM does not support feature importance
            },
            "evaluation_note": evaluation_note,
            "evaluation": evaluation,
        }
        return trained_model, report
    return trained_model


def train_and_save_isolation_forest(
    model_path: str | Path | None = None,
    return_report: bool = False,
    use_hourly_aggregation: bool = False,
    aggregation_hours: int = 24,
    aggregation_minutes: int | None = None,
    row_limit: int = LATEST_TRAINING_ROWS,
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    target_path = Path(model_path) if model_path is not None else IF_MODEL_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    training_frame = _load_training_frame(
        use_hourly_aggregation=use_hourly_aggregation,
        aggregation_hours=aggregation_hours,
        aggregation_minutes=aggregation_minutes,
        row_limit=row_limit,
        require_labels=False,
    )
    X = training_frame[FEATURE_COLUMNS]
    model, evolution = _train_isolation_forest_with_evolution(X)
    joblib.dump(model, target_path)

    predictions = model.predict(X)
    anomaly_count = int((predictions == -1).sum())
    total_count = int(len(X))
    normal_count = int(total_count - anomaly_count)
    anomaly_percentage = float((anomaly_count / total_count) * 100.0) if total_count > 0 else 0.0

    dataset_info = _build_dataset_info(
        training_frame=training_frame,
        labels=None,
        include_class_distribution=False,
    )
    model_info = {
        "name": "Isolation Forest",
        "contamination": float(model.contamination),
        "last_trained_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(target_path),
    }

    anomaly_summary = {
        "total_measurements": total_count,
        "anomaly_count": anomaly_count,
        "anomaly_percentage": anomaly_percentage,
        "contamination": float(model.contamination),
        "distribution": {
            "normal": normal_count,
            "anomaly": anomaly_count,
        },
    }
    print(
        f"Trained IsolationForest on {len(X)} latest rows from Supabase measurements "
        f"(limit={row_limit})."
    )
    print(f"Model saved to {target_path}")
    if return_report:
        report = {
            "model_type": "isolation_forest",
            "dataset_info": dataset_info,
            "model_info": model_info,
            "anomaly_summary": anomaly_summary,
            "summary": {
                "rows_requested": int(row_limit) if not use_hourly_aggregation else None,
                "rows_used": int(len(X)),
                "hourly_aggregation": bool(use_hourly_aggregation),
                "aggregation_hours": int(aggregation_hours),
                "aggregation_granularity": training_frame.attrs.get("aggregation_granularity"),
                "aggregation_value": training_frame.attrs.get("aggregation_value"),
                "source_rows_before_aggregation": (
                    int(training_frame.attrs.get("source_rows_before_aggregation"))
                    if training_frame.attrs.get("source_rows_before_aggregation") is not None
                    else None
                ),
            },
            "technical_details": {
                "evolution": evolution,
            },
            "evaluation": None,
        }
        return model, report
    return model


def load_model(
    model_path: str | Path | None = None,
    model_type: str | None = None,
    include_anomaly_models: bool = False,
):
    """
    Load a trained model. 
    
    Args:
        model_path: Explicit path to model file. If provided, loads this model.
        model_type: Specify which model to load ('random_forest', 'xgboost', 'svm', 'isolation_forest').
               If not provided, auto-detects the most recently trained classifier.
        include_anomaly_models: Include Isolation Forest in automatic model selection.
    
    Returns:
        Loaded model
    """
    if model_path is not None:
        target_path = Path(model_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found: {target_path}")
        return joblib.load(target_path)
    
    # Auto-detect the most recently trained classifier. Isolation Forest is an
    # anomaly detector and does not implement predict_proba(), which the quality
    # prediction endpoints require.
    available_models = []
    
    model_paths = {
        "random_forest": RF_MODEL_PATH,
        "xgboost": XGBOOST_MODEL_PATH,
        "svm": SVM_MODEL_PATH,
        "isolation_forest": IF_MODEL_PATH,
    }
    
    # If model_type is specified, try to load that specific model
    if model_type is not None:
        if model_type not in model_paths:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_paths.keys())}")
        target_path = model_paths[model_type]
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found for type '{model_type}': {target_path}")
        return joblib.load(target_path)
    
    # Auto-detect: find most recently modified model
    for mtype, mpath in model_paths.items():
        if mtype == "isolation_forest" and not include_anomaly_models:
            continue
        if mpath.exists():
            mtime = mpath.stat().st_mtime
            available_models.append((mtype, mpath, mtime))
    
    if not available_models:
        # Fallback to legacy model path
        legacy_model_path = MODELS_DIR / "air_quality_model.pkl"
        if legacy_model_path.exists():
            return joblib.load(legacy_model_path)
        raise FileNotFoundError(
            f"No trained models found. Available model paths: {', '.join(str(p) for p in model_paths.values())}"
        )
    
    # Use most recently modified model
    available_models.sort(key=lambda x: x[2], reverse=True)
    best_model_type, best_model_path, _ = available_models[0]
    
    return joblib.load(best_model_path)


if __name__ == "__main__":
    train_and_save_model()
