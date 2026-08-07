"""
Test suite for ModelManager and model wrappers.
"""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from app.models.model_manager import (
    ModelManager,
    RandomForestModelWrapper,
    XGBoostModelWrapper,
    SVMModelWrapper,
    IsolationForestModelWrapper,
)


@pytest.fixture
def sample_data():
    """Create sample classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        n_classes=3,
        random_state=42,
    )
    return X, y


@pytest.fixture
def sample_data_with_columns():
    """Create sample data with required feature columns."""
    X, y = make_classification(
        n_samples=100,
        n_features=9,
        n_informative=8,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    df = pd.DataFrame(
        X,
        columns=[
            "temperature",
            "humidity",
            "pressure",
            "gas",
            "lux",
            "co2",
            "pm1",
            "pm25",
            "pm10",
        ],
    )
    return df, y


class TestRandomForestWrapper:
    def test_train_and_predict(self, sample_data):
        X, y = sample_data
        model = RandomForestModelWrapper()
        assert not model.is_trained

        model.train(X, y)
        assert model.is_trained

        predictions = model.predict(X[:10])
        assert len(predictions) == 10
        assert predictions.dtype == np.int64

        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 3)  # 3 classes

    def test_save_load(self, sample_data):
        X, y = sample_data
        model = RandomForestModelWrapper()
        model.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "rf_model.pkl"
            model.save(save_path)
            assert save_path.exists()

            model2 = RandomForestModelWrapper()
            model2.load(save_path)
            assert model2.is_trained

            # Verify predictions are identical
            pred1 = model.predict(X[:5])
            pred2 = model2.predict(X[:5])
            np.testing.assert_array_equal(pred1, pred2)


class TestXGBoostWrapper:
    def test_train_and_predict(self, sample_data_with_columns):
        X, y = sample_data_with_columns
        model = XGBoostModelWrapper()
        assert not model.is_trained

        model.train(X, y)
        assert model.is_trained

        predictions = model.predict(X[:10])
        assert len(predictions) == 10

        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 2)  # 2 classes

    def test_missing_features_raises_error(self, sample_data):
        X, y = sample_data
        model = XGBoostModelWrapper()

        # X has only 5 features, XGBoost needs 9
        with pytest.raises((ValueError, KeyError, IndexError)):
            model.train(X, y)

    def test_save_load(self, sample_data_with_columns):
        X, y = sample_data_with_columns
        model = XGBoostModelWrapper()
        model.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "xgb_model.pkl"
            model.save(save_path)
            assert save_path.exists()

            model2 = XGBoostModelWrapper()
            model2.load(save_path)
            assert model2.is_trained

            # Verify predictions are identical
            pred1 = model.predict(X[:5])
            pred2 = model2.predict(X[:5])
            np.testing.assert_array_equal(pred1, pred2)


class TestSVMWrapper:
    def test_train_and_predict(self, sample_data):
        X, y = sample_data
        model = SVMModelWrapper()
        assert not model.is_trained

        model.train(X, y)
        assert model.is_trained

        predictions = model.predict(X[:10])
        assert len(predictions) == 10

        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 3)  # 3 classes

    def test_save_load(self, sample_data):
        X, y = sample_data
        model = SVMModelWrapper()
        model.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "svm_model.pkl"
            model.save(save_path)
            assert save_path.exists()

            model2 = SVMModelWrapper()
            model2.load(save_path)
            assert model2.is_trained

            # Verify predictions are identical
            pred1 = model.predict(X[:5])
            pred2 = model2.predict(X[:5])
            np.testing.assert_array_equal(pred1, pred2)


class TestIsolationForestWrapper:
    def test_train_and_predict(self, sample_data):
        X, y = sample_data  # y is not used for IF
        model = IsolationForestModelWrapper()
        assert not model.is_trained

        model.train(X)  # No y needed
        assert model.is_trained

        predictions = model.predict(X[:10])
        assert len(predictions) == 10
        assert set(predictions).issubset({-1, 1})  # -1 for anomaly, 1 for normal

        scores = model.predict_proba(X[:10])
        assert len(scores) == 10
        # Scores should be anomaly scores (negative values)
        assert np.all(scores <= 0)

    def test_save_load(self, sample_data):
        X, y = sample_data
        model = IsolationForestModelWrapper()
        model.train(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "if_model.pkl"
            model.save(save_path)
            assert save_path.exists()

            model2 = IsolationForestModelWrapper()
            model2.load(save_path)
            assert model2.is_trained

            # Verify predictions are identical
            pred1 = model.predict(X[:5])
            pred2 = model2.predict(X[:5])
            np.testing.assert_array_equal(pred1, pred2)


class TestModelManager:
    def test_available_models(self):
        manager = ModelManager()
        models = manager.get_available_models()
        assert set(models) == {"random_forest", "xgboost", "svm", "isolation_forest"}

    def test_select_model(self):
        manager = ModelManager()
        model = manager.select_model("random_forest")
        assert isinstance(model, RandomForestModelWrapper)
        assert manager.current_model == "random_forest"

    def test_select_invalid_model(self):
        manager = ModelManager()
        with pytest.raises(ValueError, match="Unknown model type"):
            manager.select_model("invalid_model")

    def test_train_random_forest(self, sample_data):
        X, y = sample_data
        manager = ModelManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            model = manager.train("random_forest", X, y, save_path=Path(tmpdir) / "rf.pkl")
            assert isinstance(model, RandomForestModelWrapper)
            assert model.is_trained
            assert (Path(tmpdir) / "rf.pkl").exists()

    def test_train_xgboost(self, sample_data_with_columns):
        X, y = sample_data_with_columns
        manager = ModelManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            model = manager.train("xgboost", X, y, save_path=Path(tmpdir) / "xgb.pkl")
            assert isinstance(model, XGBoostModelWrapper)
            assert model.is_trained
            assert (Path(tmpdir) / "xgb.pkl").exists()

    def test_train_svm(self, sample_data):
        X, y = sample_data
        manager = ModelManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            model = manager.train("svm", X, y, save_path=Path(tmpdir) / "svm.pkl")
            assert isinstance(model, SVMModelWrapper)
            assert model.is_trained
            assert (Path(tmpdir) / "svm.pkl").exists()

    def test_train_isolation_forest(self, sample_data):
        X, y = sample_data
        manager = ModelManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            model = manager.train("isolation_forest", X, save_path=Path(tmpdir) / "if.pkl")
            assert isinstance(model, IsolationForestModelWrapper)
            assert model.is_trained
            assert (Path(tmpdir) / "if.pkl").exists()

    def test_predict_with_current_model(self, sample_data):
        X, y = sample_data
        manager = ModelManager()
        manager.train("random_forest", X, y, save_model=False)

        predictions = manager.predict(X[:10])
        assert len(predictions) == 10

    def test_predict_with_specific_model(self, sample_data):
        X, y = sample_data
        manager = ModelManager()
        manager.train("random_forest", X, y, save_model=False)
        manager.train("svm", X, y, save_model=False)

        # Predict with SVM even though RF was trained last
        predictions = manager.predict(X[:10], model_type="svm")
        assert len(predictions) == 10

    def test_predict_proba(self, sample_data):
        X, y = sample_data
        manager = ModelManager()
        manager.train("random_forest", X, y, save_model=False)

        proba = manager.predict_proba(X[:10])
        assert proba.shape == (10, 3)

    def test_load_model(self, sample_data):
        X, y = sample_data
        manager = ModelManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "rf.pkl"
            manager.train("random_forest", X, y, save_path=save_path)

            # Create new manager and load
            manager2 = ModelManager()
            model = manager2.load_model("random_forest", save_path)
            assert model.is_trained

            # Verify predictions match
            pred1 = manager.predict(X[:5])
            pred2 = manager2.predict(X[:5], model_type="random_forest")
            np.testing.assert_array_equal(pred1, pred2)

    def test_is_model_trained(self, sample_data):
        X, y = sample_data
        manager = ModelManager()

        assert not manager.is_model_trained("random_forest")
        manager.train("random_forest", X, y, save_model=False)
        assert manager.is_model_trained("random_forest")

    def test_load_all_models_nonexistent(self):
        """Test loading all models when none exist."""
        manager = ModelManager()
        results = manager.load_all_models()

        # Verify results dict has all models
        assert set(results.keys()) == {"random_forest", "xgboost", "svm", "isolation_forest"}
        # At least some models should not be trained (or none if testing in isolation)
        assert isinstance(results, dict)
        assert all(isinstance(v, bool) for v in results.values())
