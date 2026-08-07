"""Test XGBoost adaptive column detection"""
import pandas as pd
import numpy as np
from app.models.xgboost_model import XGBoostModel, XGBOOST_FEATURE_COLUMNS_EXTENDED, XGBOOST_FEATURE_COLUMNS_STANDARD


def test_xgboost_with_extended_columns():
    """Test XGBoost with all 9 columns available."""
    X = pd.DataFrame(
        np.random.randn(20, 9),
        columns=XGBOOST_FEATURE_COLUMNS_EXTENDED
    )
    y = pd.Series(['good', 'poor'] * 10)
    
    model = XGBoostModel()
    model.fit(X, y)
    
    assert model.feature_columns_ == XGBOOST_FEATURE_COLUMNS_EXTENDED
    predictions = model.predict(X)
    assert len(predictions) == 20
    assert all(p in ['good', 'poor'] for p in predictions)


def test_xgboost_with_standard_columns():
    """Test XGBoost with only 5 standard columns (database columns)."""
    X = pd.DataFrame(
        np.random.randn(20, 5),
        columns=XGBOOST_FEATURE_COLUMNS_STANDARD
    )
    y = pd.Series(['good', 'poor', 'moderate'] * 7)[:20]
    
    model = XGBoostModel()
    model.fit(X, y)
    
    assert model.feature_columns_ == XGBOOST_FEATURE_COLUMNS_STANDARD
    predictions = model.predict(X)
    assert len(predictions) == 20
    assert all(p in ['good', 'poor', 'moderate'] for p in predictions)


def test_xgboost_fallback_from_extended_to_standard():
    """Test that XGBoost falls back from extended to standard columns."""
    # Provide extended columns but with some missing
    data = {
        'temperature': np.random.randn(15),
        'humidity': np.random.randn(15),
        'pm25': np.random.randn(15),
        'pm10': np.random.randn(15),
        'co2': np.random.randn(15),
        # Missing: pressure, gas, lux, pm1
    }
    X = pd.DataFrame(data)
    y = pd.Series(['good', 'poor'] * 7 + ['moderate'])
    
    model = XGBoostModel()
    model.fit(X, y)
    
    # Should fall back to standard 5 columns
    assert model.feature_columns_ == XGBOOST_FEATURE_COLUMNS_STANDARD
    predictions = model.predict(X)
    assert len(predictions) == 15


def test_xgboost_predict_proba():
    """Test predict_proba with adaptive columns."""
    X_train = pd.DataFrame(
        np.random.randn(20, 5),
        columns=XGBOOST_FEATURE_COLUMNS_STANDARD
    )
    y_train = pd.Series(['good', 'poor'] * 10)
    
    model = XGBoostModel()
    model.fit(X_train, y_train)
    
    X_test = pd.DataFrame(
        np.random.randn(5, 5),
        columns=XGBOOST_FEATURE_COLUMNS_STANDARD
    )
    proba = model.predict_proba(X_test)
    
    assert proba.shape[0] == 5
    assert proba.shape[1] == 2  # 2 classes
    assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
