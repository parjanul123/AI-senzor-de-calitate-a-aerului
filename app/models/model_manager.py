"""
ModelManager: Unified interface for managing all ML models (RF, XGBoost, SVM, IF).

Each model implements:
  - train(X, y) -> Trains the model
  - predict(X) -> Predictions
  - predict_proba(X) -> Probability predictions (anomaly scores for IF)
  - save(path) -> Serializes to disk
  - load(path) -> Deserializes from disk
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from app.core.config import MODELS_DIR, RF_MODEL_PATH, SVM_MODEL_PATH, IF_MODEL_PATH, XGBOOST_MODEL_PATH


class BaseModel(ABC):
    """Abstract base class for all ML models."""

    @abstractmethod
    def train(self, X, y) -> None:
        """Train the model on X, y."""
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Predict labels for X."""
        pass

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities/scores for X."""
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        pass

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        pass


class RandomForestModelWrapper(BaseModel):
    """Wrapper for scikit-learn RandomForestClassifier."""

    def __init__(self, **kwargs):
        default_params = {
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
            "oob_score": True,
        }
        default_params.update(kwargs)
        self.model = RandomForestClassifier(**default_params)
        self._is_trained = False

    def train(self, X, y) -> None:
        """Train Random Forest model."""
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, X) -> np.ndarray:
        """Predict using Random Forest."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities using Random Forest."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        self._is_trained = True

    @property
    def is_trained(self) -> bool:
        return self._is_trained


class XGBoostModelWrapper(BaseModel):
    """Wrapper for XGBoost classifier."""

    def __init__(self, **kwargs):
        self.classifier = None
        self.kwargs = kwargs
        self.label_encoder = LabelEncoder()
        self._is_trained = False
        self.feature_names_ = None
        self.feature_columns_ = None  # Will be detected on first call

    def _detect_available_features(self, X) -> list[str]:
        """Detect which features are available in the data."""
        extended_cols = [
            "temperature", "humidity", "pressure", "gas", "lux", "co2", "pm1", "pm25", "pm10"
        ]
        standard_cols = ["temperature", "humidity", "pm25", "pm10", "co2"]
        
        if isinstance(X, pd.DataFrame):
            available = set(X.columns)
        else:
            return standard_cols
        
        # Check extended set first
        if all(col in available for col in extended_cols):
            return extended_cols
        
        # Fall back to standard set
        if all(col in available for col in standard_cols):
            return standard_cols
        
        # Use whatever standard columns are available
        available_standard = [col for col in standard_cols if col in available]
        if available_standard:
            return available_standard
        
        # Fallback to numeric columns
        if isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                return numeric_cols
        
        raise ValueError(
            "Baza de date nu are coloane numerice suficiente pentru XGBoost. "
            "Sunt necesare cel puțin: temperature, humidity, pm25, pm10, co2"
        )

    def train(self, X, y) -> None:
        """Train XGBoost model."""
        X_prepared = self._prepare_features(X)
        labels = pd.Series(y)

        if len(X_prepared) != len(labels):
            raise ValueError("X and y must have the same number of rows.")
        if labels.nunique() < 2:
            raise ValueError("XGBoost requires at least two classes.")

        encoded_labels = self.label_encoder.fit_transform(labels)
        n_classes = len(self.label_encoder.classes_)

        # Set parameters based on number of classes
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": 1,
        }

        if n_classes == 2:
            default_params["objective"] = "binary:logistic"
            default_params["eval_metric"] = "logloss"
        else:
            default_params["objective"] = "multi:softprob"
            default_params["eval_metric"] = "mlogloss"

        default_params.update(self.kwargs)
        self.classifier = XGBClassifier(**default_params)
        self.classifier.fit(X_prepared, encoded_labels)
        self._is_trained = True

    def predict(self, X) -> np.ndarray:
        """Predict using XGBoost."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        X_prepared = self._prepare_features(X)
        encoded_predictions = self.classifier.predict(X_prepared)
        return self.label_encoder.inverse_transform(encoded_predictions.astype(int))

    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities using XGBoost."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        X_prepared = self._prepare_features(X)
        return self.classifier.predict_proba(X_prepared)

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.classifier, "label_encoder": self.label_encoder}, path)

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        data = joblib.load(path)
        
        # Handle both old format (raw model) and new format (dict with model+label_encoder)
        if isinstance(data, dict) and "model" in data:
            self.classifier = data["model"]
            self.label_encoder = data.get("label_encoder", LabelEncoder())
        else:
            # Old format: assume it's the classifier
            self.classifier = data
            self.label_encoder = LabelEncoder()
        
        self._is_trained = True

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _prepare_features(self, X) -> pd.DataFrame:
        """Prepare and validate features with adaptive column detection."""
        # Detect available features on first call
        if self.feature_columns_ is None:
            self.feature_columns_ = self._detect_available_features(X)
        
        if isinstance(X, pd.DataFrame):
            missing = [col for col in self.feature_columns_ if col not in X.columns]
            if missing:
                raise ValueError(f"Missing XGBoost features: {', '.join(missing)}")
            prepared = X.loc[:, self.feature_columns_].copy()
        else:
            if len(X) != len(self.feature_columns_):
                raise ValueError(
                    f"Shape mismatch: expected {len(self.feature_columns_)} features, got {len(X)}"
                )
            prepared = pd.DataFrame(X, columns=self.feature_columns_)

        prepared = prepared.apply(pd.to_numeric, errors="coerce")
        if prepared.isna().any().any():
            raise ValueError("XGBoost features must be numeric and non-null.")
        return prepared


class SVMModelWrapper(BaseModel):
    """Wrapper for scikit-learn SVM classifier."""

    def __init__(self, **kwargs):
        default_params = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "random_state": 42,
        }
        default_params.update(kwargs)
        self.scaler = StandardScaler()
        self.model = SVC(**default_params)
        self._is_trained = False

    def train(self, X, y) -> None:
        """Train SVM model (with scaling)."""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)

        X_scaled = self.scaler.fit_transform(X_array)
        self.model.fit(X_scaled, y)
        self._is_trained = True

    def predict(self, X) -> np.ndarray:
        """Predict using SVM."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)

        X_scaled = self.scaler.transform(X_array)
        return self.model.predict(X_scaled)

    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities using SVM."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)

        X_scaled = self.scaler.transform(X_array)
        return self.model.predict_proba(X_scaled)

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        data = joblib.load(path)
        
        # Handle both old format (raw model) and new format (dict with model+scaler)
        if isinstance(data, dict) and "model" in data:
            self.model = data["model"]
            self.scaler = data.get("scaler", StandardScaler())
        else:
            # Old format: direct model save
            self.model = data
            self.scaler = StandardScaler()
        
        self._is_trained = True

    @property
    def is_trained(self) -> bool:
        return self._is_trained


class IsolationForestModelWrapper(BaseModel):
    """Wrapper for scikit-learn IsolationForest (anomaly detection)."""

    def __init__(self, **kwargs):
        default_params = {
            "n_estimators": 100,
            "contamination": 0.05,
            "random_state": 42,
            "n_jobs": -1,
        }
        default_params.update(kwargs)
        self.model = IsolationForest(**default_params)
        self._is_trained = False

    def train(self, X, y=None) -> None:
        """Train Isolation Forest (unsupervised, y is ignored)."""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)

        self.model.fit(X_array)
        self._is_trained = True

    def predict(self, X) -> np.ndarray:
        """
        Predict anomalies: -1 for anomaly, 1 for normal.
        Returns: np.ndarray of -1 or 1
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)

        return self.model.predict(X_array)

    def predict_proba(self, X) -> np.ndarray:
        """
        Anomaly scores: negative values indicate anomalies (more negative = more anomalous).
        Returns: np.ndarray of anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)

        return self.model.score_samples(X_array)

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        self._is_trained = True

    @property
    def is_trained(self) -> bool:
        return self._is_trained


class ModelManager:
    """Central manager for all ML models."""

    MODEL_TYPES = {
        "random_forest": RandomForestModelWrapper,
        "xgboost": XGBoostModelWrapper,
        "svm": SVMModelWrapper,
        "isolation_forest": IsolationForestModelWrapper,
    }

    DEFAULT_PATHS = {
        "random_forest": RF_MODEL_PATH,
        "xgboost": XGBOOST_MODEL_PATH,
        "svm": SVM_MODEL_PATH,
        "isolation_forest": IF_MODEL_PATH,
    }

    def __init__(self):
        """Initialize ModelManager with all model types."""
        self.models: dict[str, BaseModel] = {}
        self.current_model: Optional[str] = None

        # Create wrapper instances for all models
        for model_type, wrapper_class in self.MODEL_TYPES.items():
            self.models[model_type] = wrapper_class()

    def get_available_models(self) -> list[str]:
        """Return list of available model types."""
        return list(self.MODEL_TYPES.keys())

    def select_model(self, model_type: str) -> BaseModel:
        """Select a model by type. Returns the model instance."""
        if model_type not in self.models:
            raise ValueError(
                f"Unknown model type: {model_type}. Available: {self.get_available_models()}"
            )
        self.current_model = model_type
        return self.models[model_type]

    def get_current_model(self) -> Optional[BaseModel]:
        """Get currently selected model."""
        if self.current_model is None:
            return None
        return self.models[self.current_model]

    def get_model(self, model_type: str) -> BaseModel:
        """Get a specific model without selecting it."""
        if model_type not in self.models:
            raise ValueError(
                f"Unknown model type: {model_type}. Available: {self.get_available_models()}"
            )
        return self.models[model_type]

    def train(
        self,
        model_type: str,
        X,
        y=None,
        save_model: bool = True,
        save_path: Optional[str | Path] = None,
    ) -> BaseModel:
        """
        Train a specific model.

        Args:
            model_type: Type of model (random_forest, xgboost, svm, isolation_forest)
            X: Training features
            y: Training labels (optional for isolation_forest)
            save_model: Whether to save the model after training
            save_path: Path to save the model (uses default if None)

        Returns:
            Trained model instance
        """
        model = self.select_model(model_type)
        model.train(X, y)

        if save_model:
            if save_path is None:
                save_path = self.DEFAULT_PATHS[model_type]
            model.save(save_path)

        return model

    def predict(self, X, model_type: Optional[str] = None) -> np.ndarray:
        """
        Predict using a specific model or the currently selected one.

        Args:
            X: Features to predict
            model_type: Model type to use (if None, uses current_model)

        Returns:
            Predictions
        """
        if model_type is not None:
            model = self.get_model(model_type)
        else:
            model = self.get_current_model()
            if model is None:
                raise ValueError("No model selected. Call select_model() first.")

        return model.predict(X)

    def predict_proba(self, X, model_type: Optional[str] = None) -> np.ndarray:
        """
        Predict probabilities/scores using a specific model or the currently selected one.

        Args:
            X: Features to predict
            model_type: Model type to use (if None, uses current_model)

        Returns:
            Probabilities or anomaly scores
        """
        if model_type is not None:
            model = self.get_model(model_type)
        else:
            model = self.get_current_model()
            if model is None:
                raise ValueError("No model selected. Call select_model() first.")

        return model.predict_proba(X)

    def load_model(self, model_type: str, path: Optional[str | Path] = None) -> BaseModel:
        """
        Load a trained model from disk.

        Args:
            model_type: Type of model to load
            path: Path to model file (uses default if None)

        Returns:
            Loaded model instance
        """
        model = self.select_model(model_type)

        if path is None:
            path = self.DEFAULT_PATHS[model_type]

        model.load(path)
        return model

    def save_model(self, model_type: str, path: Optional[str | Path] = None) -> None:
        """
        Save a trained model to disk.

        Args:
            model_type: Type of model to save
            path: Path to save the model (uses default if None)
        """
        model = self.get_model(model_type)

        if path is None:
            path = self.DEFAULT_PATHS[model_type]

        model.save(path)

    def is_model_trained(self, model_type: str) -> bool:
        """Check if a model has been trained."""
        model = self.get_model(model_type)
        return model.is_trained

    def load_all_models(self) -> dict[str, bool]:
        """
        Try to load all available models from their default paths.

        Returns:
            Dictionary with model_type -> load_success
        """
        results = {}
        for model_type, path in self.DEFAULT_PATHS.items():
            try:
                self.load_model(model_type, path)
                results[model_type] = True
            except FileNotFoundError:
                results[model_type] = False
        return results
