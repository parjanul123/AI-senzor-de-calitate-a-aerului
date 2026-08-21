from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from app.core.config import MODELS_DIR, XGBOOST_MODEL_PATH
from app.models.train_model import (
    DATABASE_LABEL_SOURCE,
    LATEST_TRAINING_ROWS,
    QUANTILE_LABEL_SOURCE,
    SYNTHETIC_LABEL_SOURCE,
    TARGET_COLUMN,
    _build_dataset_info,
    _build_training_summary,
    _load_training_frame,
)


# Primary feature set (9 columns) - best performance
XGBOOST_FEATURE_COLUMNS_EXTENDED = [
    "temperature",
    "humidity",
    "pressure",
    "gas",
    "lux",
    "co2",
    "pm1",
    "pm25",
    "pm10",
]

# Fallback feature set (5 columns) - compatible with existing database
XGBOOST_FEATURE_COLUMNS_STANDARD = [
    "temperature",
    "humidity",
    "pm25",
    "pm10",
    "co2",
]


class XGBoostModel:
    """XGBoost classifier with adaptive feature columns and same fit/predict/predict_proba API."""

    def __init__(self, **classifier_params):
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": 1,
        }
        default_params.update(classifier_params)
        self.classifier = None
        self.classifier_params = default_params
        self.label_encoder = LabelEncoder()
        self.feature_columns_ = None  # Will be set during fit
        self.classes_ = np.array([], dtype=object)

    def _detect_available_features(self, features) -> list[str]:
        """Detect which features are available in the data."""
        if isinstance(features, pd.DataFrame):
            available_cols = set(features.columns)
        else:
            # If it's not a DataFrame, we can't detect, so try standard first
            return XGBOOST_FEATURE_COLUMNS_STANDARD
        
        # Check extended set first (9 columns)
        extended_available = [col for col in XGBOOST_FEATURE_COLUMNS_EXTENDED if col in available_cols]
        if len(extended_available) == len(XGBOOST_FEATURE_COLUMNS_EXTENDED):
            return XGBOOST_FEATURE_COLUMNS_EXTENDED
        
        # Fall back to standard set (5 columns)
        standard_available = [col for col in XGBOOST_FEATURE_COLUMNS_STANDARD if col in available_cols]
        if len(standard_available) == len(XGBOOST_FEATURE_COLUMNS_STANDARD):
            return XGBOOST_FEATURE_COLUMNS_STANDARD
        
        # If we're missing standard columns too, use whatever is available
        if standard_available:
            return standard_available
        
        # Fallback to numeric columns only
        if isinstance(features, pd.DataFrame):
            numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                return numeric_cols
        
        raise ValueError(
            "Baza de date nu are coloane numerice suficiente pentru XGBoost. "
            "Sunt necesare cel puțin: temperature, humidity, pm25, pm10, co2"
        )

    def _prepare_features(self, features) -> pd.DataFrame:
        """Prepare features using detected available columns."""
        if self.feature_columns_ is None:
            # First time: detect what's available
            self.feature_columns_ = self._detect_available_features(features)
        
        if isinstance(features, pd.DataFrame):
            missing_columns = [col for col in self.feature_columns_ if col not in features.columns]
            if missing_columns:
                raise ValueError(
                    f"Lipsesc coloane pentru XGBoost: {', '.join(missing_columns)}"
                )
            prepared = features.loc[:, self.feature_columns_].copy()
        else:
            if len(features) != len(self.feature_columns_):
                raise ValueError(
                    f"Shape mismatch: expected {len(self.feature_columns_)} features, "
                    f"got {len(features) if hasattr(features, '__len__') else '?'}"
                )
            prepared = pd.DataFrame(features, columns=self.feature_columns_)

        prepared = prepared.apply(pd.to_numeric, errors="coerce")
        if prepared.isna().any().any():
            raise ValueError("Feature-urile XGBoost trebuie să fie numerice și nenule.")
        return prepared

    def fit(self, X, y):
        features = self._prepare_features(X)
        labels = pd.Series(y)
        if len(features) != len(labels):
            raise ValueError("X și y trebuie să aibă același număr de rânduri.")
        if labels.nunique() < 2:
            raise ValueError("XGBoost necesită cel puțin două clase pentru antrenare.")

        encoded_labels = self.label_encoder.fit_transform(labels)
        self.classes_ = self.label_encoder.classes_
        
        # Determine objective and eval_metric based on number of classes
        n_classes = len(self.label_encoder.classes_)
        if n_classes == 2:
            self.classifier_params["objective"] = "binary:logistic"
            self.classifier_params["eval_metric"] = "logloss"
        else:
            self.classifier_params["objective"] = "multi:softprob"
            self.classifier_params["eval_metric"] = "mlogloss"
        
        # Initialize classifier with proper parameters
        self.classifier = XGBClassifier(**self.classifier_params)
        self.classifier.fit(features, encoded_labels)
        return self

    def predict(self, X):
        features = self._prepare_features(X)
        encoded_predictions = self.classifier.predict(features)
        return self.label_encoder.inverse_transform(encoded_predictions.astype(int))

    def predict_proba(self, X):
        features = self._prepare_features(X)
        return self.classifier.predict_proba(features)


def load_xgboost_training_data(
    use_hourly_aggregation: bool = False,
    aggregation_hours: int = 24,
    aggregation_minutes: int | None = None,
    row_limit: int = LATEST_TRAINING_ROWS,
    allow_derived_label_fallback: bool = False,
    return_metadata: bool = False,
    device_identifier: str | None = None,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, object]]:
    """Load labeled measurements through the existing shared training loader."""
    training_frame = _load_training_frame(
        use_hourly_aggregation=use_hourly_aggregation,
        aggregation_hours=aggregation_hours,
        aggregation_minutes=aggregation_minutes,
        row_limit=row_limit,
        require_labels=True,
        allow_derived_label_fallback=allow_derived_label_fallback,
        device_identifier=device_identifier,
    )
    
    # Adaptive column detection
    available_cols = set(training_frame.columns)
    
    # Try extended set first
    feature_columns = None
    if all(col in available_cols for col in XGBOOST_FEATURE_COLUMNS_EXTENDED):
        feature_columns = XGBOOST_FEATURE_COLUMNS_EXTENDED
    # Fall back to standard set
    elif all(col in available_cols for col in XGBOOST_FEATURE_COLUMNS_STANDARD):
        feature_columns = XGBOOST_FEATURE_COLUMNS_STANDARD
    else:
        # Last resort: use numeric columns that are available from standard set
        available_standard = [col for col in XGBOOST_FEATURE_COLUMNS_STANDARD if col in available_cols]
        if available_standard:
            feature_columns = available_standard
        else:
            raise RuntimeError(
                "Lipsesc coloane necesare pentru XGBoost din tabela 'measurements'. "
                "Sunt necesare cel puțin: temperature, humidity, pm25, pm10, co2"
            )

    X = training_frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    valid_rows = X.notna().all(axis=1)
    X = X.loc[valid_rows].copy()
    y = training_frame.loc[X.index, TARGET_COLUMN].copy()
    filtered_training_frame = training_frame.loc[X.index].copy()
    if X.empty:
        raise RuntimeError("Nu există rânduri valide pentru antrenarea XGBoost.")
    if y.nunique() < 2:
        raise RuntimeError("XGBoost necesită cel puțin două clase pentru antrenare.")

    if not return_metadata:
        return X, y

    metadata = {
        "label_source": str(training_frame.attrs.get("label_source", DATABASE_LABEL_SOURCE)),
        "hourly_aggregation": bool(training_frame.attrs.get("hourly_aggregation", False)),
        "aggregation_granularity": training_frame.attrs.get("aggregation_granularity"),
        "aggregation_value": training_frame.attrs.get("aggregation_value"),
        "effective_row_limit": training_frame.attrs.get("effective_row_limit"),
        "source_rows_before_aggregation": training_frame.attrs.get("source_rows_before_aggregation"),
    }
    return X, y, filtered_training_frame, metadata


def _evaluate_xgboost_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> dict | None:
    """Evaluate XGBoost model using holdout test set."""
    labels = sorted(y.unique().tolist())

    if len(X) < 10:
        return None

    group_frame = X.copy()
    group_frame["__target"] = y.astype(str)
    row_groups = group_frame.astype(str).agg("|".join, axis=1)

    if row_groups.nunique() < 2:
        return None

    TEST_SIZE = 0.2
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

    # Fit a fresh model on training data for evaluation
    eval_model = XGBoostModel().fit(X_train, y_train)
    y_pred = eval_model.predict(X_test)
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


def train_and_save_xgboost(
    model_path: str | Path | None = None,
    return_report: bool = False,
    use_hourly_aggregation: bool = False,
    aggregation_hours: int = 24,
    aggregation_minutes: int | None = None,
    row_limit: int = LATEST_TRAINING_ROWS,
    allow_derived_label_fallback: bool = False,
    device_identifier: str | None = None,
) -> tuple[XGBoostModel, dict] | XGBoostModel:
    """Train XGBoost on measurements and persist it to models/xgboost.pkl by default."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = Path(model_path) if model_path is not None else XGBOOST_MODEL_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    X, y, training_frame, metadata = load_xgboost_training_data(
        use_hourly_aggregation=use_hourly_aggregation,
        aggregation_hours=aggregation_hours,
        aggregation_minutes=aggregation_minutes,
        row_limit=row_limit,
        allow_derived_label_fallback=allow_derived_label_fallback,
        return_metadata=True,
        device_identifier=device_identifier,
    )
    model = XGBoostModel().fit(X, y)
    joblib.dump(model, target_path)
    print(f"Trained XGBoostModel on {len(X)} rows from Supabase measurements.")
    print(f"Model saved to {target_path}")

    if not return_report:
        return model

    evaluation = None
    label_source = str(metadata.get("label_source", DATABASE_LABEL_SOURCE))
    if label_source == DATABASE_LABEL_SOURCE:
        evaluation = _evaluate_xgboost_model(X, y)

    # Extract feature importance
    feature_importances = None
    try:
        importances = model.classifier.feature_importances_
        if model.feature_columns_ is not None:
            feature_importances = {
                model.feature_columns_[i]: float(importances[i])
                for i in range(min(len(model.feature_columns_), len(importances)))
            }
    except (AttributeError, IndexError, TypeError):
        pass

    hourly_aggregation = bool(metadata.get("hourly_aggregation", use_hourly_aggregation))
    aggregation_granularity = metadata.get("aggregation_granularity")
    aggregation_value = metadata.get("aggregation_value")
    requested_rows = metadata.get("effective_row_limit")
    source_rows_before_aggregation = metadata.get("source_rows_before_aggregation")

    dataset_info = _build_dataset_info(
        training_frame=training_frame,
        labels=y,
        include_class_distribution=True,
    )
    model_info = {
        "name": "XGBoost Classifier",
        "n_estimators": int(model.classifier.n_estimators),
        "last_trained_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(target_path),
    }

    evaluation_note = None
    if label_source in {SYNTHETIC_LABEL_SOURCE, QUANTILE_LABEL_SOURCE}:
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

    if evaluation:
        recommended_metric = {
            "type": "f1_score",
            "label": "F1-score",
            "f1_score": evaluation.get("f1_score"),
        }
    else:
        # No holdout evaluation available (fallback labels or too few rows);
        # report an in-sample training score so the UI still shows a number instead of N/A.
        y_train_pred = model.predict(X)
        recommended_metric = {
            "type": "f1_score_training",
            "label": "F1-score (pe setul de antrenare)",
            "f1_score": float(f1_score(y, y_train_pred, average="weighted", zero_division=0)),
            "iteration_count": int(model.classifier.n_estimators),
        }

    summary = _build_training_summary(
        y,
        label_source=label_source,
        hourly_aggregation=hourly_aggregation,
        aggregation_hours=aggregation_hours,
        aggregation_granularity=aggregation_granularity,
        aggregation_value=int(aggregation_value) if aggregation_value is not None else None,
        row_limit=int(requested_rows) if requested_rows is not None else None,
        source_rows_before_aggregation=(
            int(source_rows_before_aggregation) if source_rows_before_aggregation is not None else None
        ),
    )

    report = {
        "model_type": "xgboost",
        "dataset_info": dataset_info,
        "model_info": model_info,
        "summary": summary,
        "technical_details": {
            "feature_importances": feature_importances,
        },
        "evaluation_note": evaluation_note,
        "evaluation": evaluation,
        "recommended_metric": recommended_metric,
    }

    return model, report